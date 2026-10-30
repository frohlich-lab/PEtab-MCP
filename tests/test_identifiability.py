"""Unit tests for the identifiability statistics core (:func:`identifiability._identifiability_stats`).

These exercise the pure math — FIM -> SE / marginal & partial correlations / flat eigen-directions
/ candidates, plus the multistart-spread signal — with hand-built matrices, so no AMICI/pyPESTO
fit is needed. The pyPESTO glue (`_compute_fim`, `identifiability`) is covered by the tests that
run a real fit, not here.
"""

from __future__ import annotations

import numpy as np
import pytest

from petab_mcp.identifiability import _identifiability_stats, identifiability


def _meta(*ids, scale="log10", est_scaled=1.0, at_bound=False):
    return [
        {
            "id": i,
            "scale": scale,
            "estimate": 10.0,
            "estimate_scaled": est_scaled,
            "at_bound": at_bound,
        }
        for i in ids
    ]


class _StubObjective:
    """What ``_compute_fim`` FALLS BACK to when an objective has no residual mode (this one takes
    no ``mode``, so the row-built FIM is unreachable through it): a Hessian, or an sres to square,
    at a point. Substituted so the ASSEMBLY around the FIM — the record, the coordinate order, the
    staleness label — is testable without an AMICI build."""

    def __init__(self, hess=None, sres=None):
        self.hess, self.sres = hess, sres
        self.calls: list[tuple] = []

    def __call__(self, x, sensi_orders=(0,), return_dict=False):
        self.calls.append((tuple(np.asarray(x, float)), sensi_orders))
        if sensi_orders == (0, 1, 2):
            if self.hess is None:
                raise RuntimeError("order-2 sensitivities unsupported for this objective")
            return {"fval": 1.0, "hess": self.hess}
        return {"fval": 1.0, "sres": self.sres}


class _StubProblem:
    def __init__(self, objective, x_names):
        self.objective = objective
        self.x_names = list(x_names)


def _record(
    session,
    *,
    names=("k1", "k2"),
    x=(0.0, -1.0),
    hess=None,
    sres=None,
    scales=("log10", "log10"),
    lb=(-4.0, -4.0),
    ub=(2.0, 2.0),
    objective=None,
    hierarchical=False,
    optimization=None,
    structure_hash=None,
    problem_hash=None,
):
    """A :class:`fitting.FitRecord` over the toy problem, with the FIM source stubbed."""
    from petab_mcp.fitting import FitRecord
    from petab_mcp.parameters import structure_fingerprint

    return FitRecord(
        pypesto_problem=_StubProblem(objective or _StubObjective(hess=hess, sres=sres), names),
        problem=session.problem,
        structure_hash=structure_hash
        or structure_fingerprint(session.problem.model.sbml_model, session.problem.parameter_df),
        problem_hash=problem_hash or session.problem_hash(),
        x_names=tuple(names),
        x_free=tuple(x),
        scales=tuple(scales),
        lb=tuple(lb),
        ub=tuple(ub),
        optimization=optimization,
        hierarchical=hierarchical,
    )


def test_collinear_pair_flagged_via_singular_fim():
    # a,b perfectly collinear (rank-1 block); c well determined.
    fim = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 4.0]])
    out = _identifiability_stats(_meta("a", "b", "c"), fim, {}, corr_threshold=0.9)

    assert out["available"] and out["method"] == "fim"
    assert out["n_estimated"] == 3
    assert out["fim"]["rank"] == 2 and out["fim"]["rank_deficient"] is True

    # marginal correlation a-b ~ +1
    ab = [c for c in out["correlations"] if {c["a"], c["b"]} == {"a", "b"}]
    assert ab and ab[0]["r"] == pytest.approx(1.0, abs=1e-6)
    # partial correlation a-b ~ -1 (from the precision/FIM directly)
    pab = [p for p in out["partial_correlations"] if {p["a"], p["b"]} == {"a", "b"}]
    assert pab and pab[0]["pr"] == pytest.approx(-1.0, abs=1e-6)

    # the a-b difference is a flat direction
    assert out["flat_directions"]
    flat_members = {p["id"] for d in out["flat_directions"] for p in d["participants"]}
    assert {"a", "b"} <= flat_members

    # a and b are simplification candidates; c is not
    cand = {c["id"] for c in out["candidates"]}
    assert {"a", "b"} <= cand and "c" not in cand
    # collinearity reason names the partner
    reasons_a = next(c["reasons"] for c in out["candidates"] if c["id"] == "a")
    assert any("collinear_with:b" in r for r in reasons_a)


def test_well_identified_has_no_flags():
    fim = np.diag([4.0, 4.0, 4.0])
    out = _identifiability_stats(_meta("a", "b", "c"), fim, {})
    assert out["available"] and out["method"] == "fim"
    assert out["correlations"] == [] and out["partial_correlations"] == []
    assert out["flat_directions"] == []
    assert out["candidates"] == []
    assert out["fim"]["rank"] == 3 and out["fim"]["rank_deficient"] is False
    # SE = sqrt(1/4) = 0.5 on the scaled axis; fold = 10**0.5 for a log10 parameter
    p = {x["id"]: x for x in out["parameters"]}
    assert p["a"]["se_scaled"] == pytest.approx(0.5, abs=1e-6)
    assert p["a"]["fold_uncertainty"] == pytest.approx(10**0.5, rel=1e-3)


def test_multistart_spread_flags_unstable_parameter():
    fim = np.diag([4.0, 4.0])
    spread = {
        "a": {"std": 0.7, "min": -0.5, "max": 1.5, "n": 5},
        "b": {"std": 0.01, "min": 0.99, "max": 1.01, "n": 5},
    }
    out = _identifiability_stats(_meta("a", "b"), fim, spread, spread_unstable=0.5)
    assert out["method"] == "fim+multistart"
    p = {x["id"]: x for x in out["parameters"]}
    assert p["a"]["multistart_std"] == 0.7 and p["a"]["multistart_range"] == pytest.approx(2.0)
    cand = {c["id"] for c in out["candidates"]}
    assert "a" in cand and "b" not in cand
    assert any(
        "multistart_unstable" in r
        for r in next(c["reasons"] for c in out["candidates"] if c["id"] == "a")
    )


def test_spread_only_when_no_fim():
    spread = {"a": {"std": 0.9, "min": 0.0, "max": 2.0, "n": 4}}
    out = _identifiability_stats(_meta("a", "b"), None, spread)
    assert out["method"] == "multistart"
    assert out["fim"] is None
    assert out["correlations"] == [] and out["flat_directions"] == []
    assert {x["id"]: x for x in out["parameters"]}["a"]["se_scaled"] is None
    assert {c["id"] for c in out["candidates"]} == {"a"}


def test_at_bound_is_a_candidate():
    fim = np.diag([4.0, 4.0])
    meta = _meta("a", "b")
    meta[0]["at_bound"] = True
    out = _identifiability_stats(meta, fim, {})
    a = next(c for c in out["candidates"] if c["id"] == "a")
    assert "at_bound" in a["reasons"]
    assert "b" not in {c["id"] for c in out["candidates"]}


def test_no_signal_reports_unavailable():
    out = _identifiability_stats(_meta("a"), None, {})
    assert out["available"] is False and out["method"] == "none"
    assert out["candidates"] == []


def test_an_unevaluable_objective_degrades_to_the_spread(session):
    """A FIM failure drops the correlation half and keeps the rest — the free parameters are the
    RECORD's either way, so the payload's coordinate list never depends on the FIM succeeding."""
    session.fit_record = _record(session, hess=None, sres=None)  # neither order works

    out = identifiability(session)

    assert out["fim"] is None and out["merge_candidates"] is None
    assert [p["id"] for p in out["parameters"]] == ["k1", "k2"]


# -- which FIM the diagnosis inverts ---------------------------------------
class _SigmaFlag:
    """An AMICI model reduced to the flag :func:`identifiability._fim_from_rows` toggles."""

    def __init__(self):
        self.sigma_residuals = False
        self.flags_seen: list[bool] = []

    def get_add_sigma_residuals(self):
        return self.sigma_residuals

    def set_add_sigma_residuals(self, flag):
        self.sigma_residuals = bool(flag)


class _RowsObjective:
    """A built objective in the shape the row-built FIM is driven through: residual mode with
    AMICI's sigma pseudo-residuals, an AMICI model whose flag it toggles — and a DIFFERENT `hess`,
    so which of the two the diagnosis inverts is observable rather than inferred."""

    def __init__(self, res, sres, blocks, hess):
        self.amici_model = _SigmaFlag()
        self.res, self.sres, self.blocks, self.hess = res, sres, blocks, hess

    def __call__(self, x, sensi_orders=(0,), mode="mode_fun", return_dict=False):
        if mode != "mode_res":
            return {"fval": 1.0, "hess": self.hess}
        self.amici_model.flags_seen.append(self.amici_model.sigma_residuals)
        return {
            "res": self.res,
            "sres": self.sres,
            "rdatas": [{"res": [0.0] * b} for b in self.blocks],
        }


def _sigma_objective():
    """One estimated parameter, ln σ, on four residuals at the MLE of a shared sigma. Its Fisher
    information is 2 per point, so the row-built FIM is 2n where the `hess` AMICI reports — the
    data rows' sres alone — is n. Same construction as
    tests/test_effective_parameter_count.py::test_an_estimated_sigma_carries_its_full_fisher_information.
    """
    rho, n = 3.0, 4
    r = np.sqrt([1.5, 1.5, 0.5, 0.5]) * [1, -1, 1, -1]  # Σ r² = n
    res = np.concatenate([r, np.full(n, rho)])
    sres = np.concatenate([-r, np.full(n, 1.0 / rho)])[:, None]  # ∂r/∂lnσ = −r, ∂ρ/∂lnσ = 1/ρ
    hess = float((r**2).sum())
    return _RowsObjective(res, sres, [2 * n], np.array([[hess]])), hess


def test_the_diagnosis_inverts_the_row_built_fim_not_the_objectives_hessian(session):
    """REGRESSION: `hess` leaves an ESTIMATED sigma at HALF its Fisher information (AMICI drops the
    term it documents dropping — docs/RATIONALE.md), and se_scaled / fold_uncertainty / the
    correlations / every merge price are read off the INVERSE of this matrix, so each was reported
    against a sigma the data pin twice as tightly as the matrix said. It is also the same matrix
    `effective_parameter_count` divides by — one construction, one evaluation, one answer."""
    from petab_mcp.identifiability import _compute_fim, effective_parameter_count

    obj, hess = _sigma_objective()
    record = _record(
        session, names=("sigma_B",), x=(0.0,), scales=("lin",), lb=(-4.0,), ub=(4.0,), objective=obj
    )

    fim = _compute_fim(record)

    assert fim == pytest.approx(np.array([[2.0 * hess]]))  # 2n, not the n `hess` reports
    scores = obj.res[:4] * obj.sres[:4, 0] + obj.res[4:] * obj.sres[4:, 0]
    assert effective_parameter_count(record.pypesto_problem, [0.0]) == pytest.approx(
        float((scores**2).sum()) / float(fim[0, 0])
    )


def test_the_sigma_residual_flag_is_restored_after_a_diagnosis(session):
    """The diagnosis flips it on the objective the ENGINE simulates through — one pyPESTO problem
    serves the fit and every evaluation (`fitting.estimation_problem`) — so leaving it on would
    double the residual vector of every later simulation."""
    obj, _ = _sigma_objective()
    session.fit_record = _record(
        session, names=("sigma_B",), x=(0.0,), scales=("lin",), lb=(-4.0,), ub=(4.0,), objective=obj
    )

    assert identifiability(session)["fim"]["rank"] == 1

    assert obj.amici_model.flags_seen == [True]
    assert obj.amici_model.sigma_residuals is False


def test_an_objective_with_no_residual_mode_still_gets_a_fim(session):
    """The rows are PREFERRED, not required: the fallbacks stay, in their order, for an objective
    that cannot answer in residual mode at all."""
    from petab_mcp.identifiability import _compute_fim

    hess = np.array([[4.0, 0.0], [0.0, 1.0]])
    assert _compute_fim(_record(session, hess=hess)) == pytest.approx(hess)
    sres = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    assert _compute_fim(_record(session, sres=sres)) == pytest.approx(sres.T @ sres)


# -- merge candidates: which pairs could take ONE SHARED value --------------
def test_merge_cost_is_exact_for_a_quadratic():
    """The statistic is (c@theta)^2 / (2 c@inv(H)@c) with c = e_i - e_j, so for a quadratic objective
    the prediction is not an approximation and can be asserted exactly."""
    import numpy as np

    from petab_mcp.identifiability import merge_candidates

    # H = I, theta = (1, 0): c@theta = 1, c@inv(H)@c = 2 -> cost 1/4
    rep = merge_candidates(np.eye(2), ["a", "b"], [1.0, 0.0])
    assert rep["ties"][0]["dnll"] == 0.25
    assert rep["ties"][0]["move_scaled"] == 0.5  # both meet in the middle of the gap they close
    # H = 2I, theta = (3, 1): c@theta = 2, c@inv(H)@c = 1 -> cost 2
    rep = merge_candidates(2 * np.eye(2), ["a", "b"], [3.0, 1.0])
    assert rep["ties"][0]["dnll"] == 2.0
    # two parameters already equal are free to tie, whatever the curvature
    rep = merge_candidates(np.eye(2) * 7, ["a", "b"], [1.5, 1.5])
    assert rep["ties"][0]["dnll"] == 0.0


def test_merge_candidates_rank_the_cheap_tie_first_and_price_it():
    """A near-collinear pair at nearly equal values is the cheap merge; the expensive one is the pair
    the data pins apart. The BIC price is ln(n)/2 — what dropping one parameter has to beat."""
    import math

    import numpy as np

    from petab_mcp.identifiability import merge_candidates

    H = np.array([[100.0, 99.0, 0.0], [99.0, 100.0, 0.0], [0.0, 0.0, 100.0]])
    theta = [1.0, 1.02, 3.0]  # a,b nearly equal and collinear; c far away
    rep = merge_candidates(H, ["a", "b", "c"], theta, n_data=252)
    assert rep["ties"][0]["parameters"] == ["a", "b"]
    assert rep["parameter_price_nll"] == round(math.log(252) / 2, 3)
    assert rep["ties"][0]["merge_improves_score"] is True  # cheaper than the parameter it drops
    assert rep["ties"][-1]["merge_improves_score"] is False  # the data pins these apart
    assert rep["n_merges_improving_score"] == 1


def test_merge_flag_and_dscore_agree_with_each_other():
    """The row's subject is the MERGE: `merge_improves_score` is true exactly when `dscore` is negative.
    A flag whose sense the reader has to derive gets read backwards — an agent selecting the true
    rows would merge every pair the data separates and skip every free one."""
    import math

    import numpy as np

    from petab_mcp.identifiability import merge_candidates

    # H = I so c@inv(H)@c = 2 and the cost is (theta_a - theta_b)^2 / 4: pick the gap for each side.
    expensive = merge_candidates(np.eye(2), ["a", "b"], [math.sqrt(40.0), 0.0], n_data=252)["ties"][
        0
    ]
    assert expensive["dnll"] == 10.0 and expensive["dscore"] > 0
    assert expensive["merge_improves_score"] is False  # +14.5 bic: do NOT merge

    free = merge_candidates(np.eye(2), ["a", "b"], [math.sqrt(4 * 0.022), 0.0], n_data=252)["ties"][
        0
    ]
    assert free["dnll"] == 0.022 and free["dscore"] < 0
    assert free["merge_improves_score"] is True  # -5.5 bic: merge it


def test_a_singular_fim_prices_its_free_merge_at_zero_and_first():
    """The case the whole report exists for. Where H is singular ALONG the merge direction the data
    never determined it, so tying the pair is FREE — inv(H) diverges there and the cost vanishes.
    A pseudo-inverse inverts that limit (1/lambda -> 0 for the truncated eigenvalue), collapsing the
    denominator to a rounding artifact and pricing the one free merge at ~1e17, ranked LAST."""
    import numpy as np

    from petab_mcp.identifiability import merge_candidates

    # a,b exactly collinear (the rank_deficient state fim_info already reports); c determined. Listed
    # c FIRST so the free merge is the LAST pair enumerated: "ranked first" must be the cost talking,
    # not the parameter order — three rows that all display 0.0 make index order look like a ranking.
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]])
    rep = merge_candidates(H, ["c", "a", "b"], [100.0, 1.0, 3.0], n_data=252)
    assert rep["ties"][0]["parameters"] == ["a", "b"]  # ranked FIRST, not last
    assert rep["ties"][0]["dnll"] == 0.0 and rep["ties"][0]["merge_improves_score"] is True
    # tying c to either of them is free too — the flat direction pays for it — but only by moving a
    # parameter the whole 99 there, and that is what ranks the collinear pair above them: among ties
    # the price cannot separate, the writable edit is the one with the smaller move.
    assert rep["ties"][0]["move_scaled"] == 1.0  # half the gap it closes
    assert all(t["dnll"] == 0.0 and t["move_scaled"] > 90 for t in rep["ties"][1:])

    # and with no curvature anywhere every tie is free, not unreportable
    flat = merge_candidates(np.zeros((2, 2)), ["a", "b"], [1.0, 2.0], n_data=252)
    assert flat["ties"][0]["dnll"] == 0.0


def _flat_along(w, lams=(90.0, 30.0, 10.0)):
    """A 4x4 FIM with one EXACT null direction whose overlap with ``e_a - e_b`` is ~``w``: the knob
    that decides how much of the merge the data never determined."""
    raw = np.array([w / 2.0, -w / 2.0, 1.0, 1.0])
    u = raw / np.linalg.norm(raw)
    q, _ = np.linalg.qr(np.column_stack([u, np.eye(4)[:, 1:]]))
    b = q[:, 1:] * np.sign(q[:, 0] @ u)  # an orthonormal basis of everything BUT u
    return (b * np.array(lams)) @ b.T


def test_the_merge_price_is_continuous_as_a_direction_goes_flat():
    """No threshold on an eigenvector component may decide a verdict. Truncating the flat direction
    (a pinv) prices the free merge at ~1e17; truncating the pair's COMPONENT in it instead — testing
    it against any floor — moves the same cliff rather than removing it, so a merge whose honest
    cost is 0 still lands last in the list, flagged do-not-merge, one rounding step away from free."""
    from petab_mcp.identifiability import merge_candidates

    def price(w):
        rep = merge_candidates(_flat_along(w), list("abcd"), [1.0, 3.0, 2.0, 2.5], n_data=252)
        return next(t for t in rep["ties"] if t["parameters"] == ["a", "b"])

    determined = price(0.0)["dnll"]  # c wholly inside what the data pinned down
    assert determined == 18.0
    prices = [price(w)["dnll"] for w in (0.7071, 1e-2, 1e-3, 1e-4, 1e-5)]
    assert prices == sorted(prices)  # less overlap with the flat direction, dearer
    assert all(p <= determined for p in prices)  # ...and never dearer than no overlap at all
    for cheap, dear in zip(prices[1:], prices[2:]):  # a DECADE of overlap is two orders of price
        assert dear <= 150 * cheap, (cheap, dear)  # no cliff: this ratio was 1.2 million

    # the cheap end of that ramp is cheap because the optimum wanders down the flat direction, and
    # the row says how far: a merge that needs a move of 20000 is not an edit that survives a refit.
    nearly_free = price(1e-4)
    assert nearly_free["merge_improves_score"] is True and nearly_free["move_scaled"] > 1e4
    assert price(0.7071)["move_scaled"] < 5  # ...where a real overlap needs none


def test_merge_candidates_only_offers_ties_within_a_role():
    """Nothing mathematically stops tying a kinetic rate to a noise sigma, and the arithmetic would
    call it free — but it is not a model edit anyone would write. An UNKNOWN role is not a role: two
    parameters missing from the table must not match each other."""
    import numpy as np

    from petab_mcp.identifiability import merge_candidates

    names = ["k1", "k2", "sigma"]
    roles = {"k1": "kinetic", "k2": "kinetic", "sigma": "noise_sigma"}
    rep = merge_candidates(np.eye(3), names, [1.0, 1.0, 1.0], roles=roles)
    assert rep["n_ties_priced"] == 1  # k1=k2 only, not either with sigma
    assert rep["ties"][0]["parameters"] == ["k1", "k2"]

    assert rep["n_ties_skipped"] == {"role_mismatch": 2}  # and it says so, by reason

    # k2 and sigma are both absent from the table: unknown == unknown is not a match, and k1=k2
    # (whose roles cannot be compared) is not offered either. "I cannot tell" is not "they differ".
    partial = merge_candidates(np.eye(3), names, [1.0, 1.0, 1.0], roles={"k1": "kinetic"})
    assert partial["ties"] == [] and partial["n_ties_skipped"] == {"role_unknown": 3}

    # no role table at all: the offers widen, and the payload SAYS so rather than pretending
    off = merge_candidates(np.eye(3), names, [1.0, 1.0, 1.0])
    assert off["n_ties_priced"] == 3 and "role" in off["filters_off"]


def test_merge_candidates_only_offer_ties_within_a_kind():
    """PEtab-MCP#167: `make_changes` refuses an id shared across kinds, so a hint tying a transition
    rate to a marker's decay, both `kinetic`, asked for an edit it then refused. An id with no kind
    (a sigma, a start the trace made) is left to the role filter."""
    import numpy as np

    from petab_mcp.identifiability import merge_candidates

    names = ["t1_k", "t2_k", "m_kd", "s"]
    transition = ("PopulationTransition", "mass_action", "k")
    kinds = {"t1_k": transition, "t2_k": transition, "m_kd": ("IncreaseAmount", "mean_approach", "kd")}
    rep = merge_candidates(
        np.eye(4), names, [1.0] * 4, roles=dict.fromkeys(names, "kinetic"), kinds=kinds
    )
    priced = [set(t["parameters"]) for t in rep["ties"]]
    assert {"t1_k", "t2_k"} in priced
    assert {"t1_k", "m_kd"} not in priced and {"t2_k", "m_kd"} not in priced
    assert rep["n_ties_priced"] == 4 and rep["n_ties_skipped"] == {"kind_mismatch": 2}


def test_merge_candidates_refuse_pairs_they_cannot_price_honestly():
    """Fail closed. Across parameterScales the equality is meaningless (`log10(k_a) == k_b` is not
    an edit anyone writes), and at a bound the quadratic's interior-stationary-point assumption is
    gone, so the cost would be UNDERSTATED and the pair offered as near-free."""
    import numpy as np

    from petab_mcp.identifiability import merge_candidates

    names, roles = ["k_a", "k_b"], {"k_a": "kinetic", "k_b": "kinetic"}
    mixed = merge_candidates(np.eye(2), names, [1.0, 1.0], scales=["log10", "lin"], roles=roles)
    assert mixed["ties"] == [] and mixed["n_ties_skipped"] == {"scale_mismatch": 1}
    same = merge_candidates(np.eye(2), names, [1.0, 1.0], scales=["log10", "log10"], roles=roles)
    assert same["n_ties_priced"] == 1 and same["filters_off"] == ["at_bound"]

    unusable = merge_candidates(
        np.eye(2), names, [1.0, 1.0], scales=[float("nan")] * 2, roles=roles
    )
    assert unusable["n_ties_skipped"] == {"scale_unknown": 1}  # unverifiable is not a match either

    pinned = merge_candidates(np.eye(2), names, [1.0, 1.0], at_bound=[True, False], roles=roles)
    assert pinned["ties"] == [] and pinned["n_ties_skipped"] == {"at_bound": 1}


def test_merge_candidates_rank_free_ties_by_cost_not_enumeration_order():
    """Ranking on the rounded number would leave the cheapest merges in whatever order the loop
    happened to visit them, and a cost rounded to 0.0 reads as "the data never determined this"
    when it was 4e-4 — so a positive cost keeps a digit instead of collapsing."""
    import numpy as np

    from petab_mcp.identifiability import merge_candidates

    # H = I: cost is (theta_i - theta_j)^2 / 4, so d=c is the cheapest pair and a=b the dearest,
    # and enumeration order (a,b) first is exactly backwards.
    theta = [0.0, 0.04, 0.0283, 0.02]
    rep = merge_candidates(np.eye(4), ["a", "b", "c", "d"], theta, top_n=6)
    assert [t["parameters"] for t in rep["ties"]][:2] == [["c", "d"], ["b", "c"]]
    assert rep["ties"][-1]["parameters"] == ["a", "b"]
    assert [t["dnll"] for t in rep["ties"]] == [2e-05, 3e-05, 0.0001, 0.0001, 0.0002, 0.0004]


def test_merge_candidates_is_bounded_and_degrades_quietly():
    """It rides in a tool payload, so the list is capped, the JSON stays Inf-free, and a shape it
    cannot use is None rather than an exception — this is advisory next to a fit."""
    import json

    import numpy as np

    from petab_mcp.identifiability import merge_candidates

    rep = merge_candidates(np.eye(8), [f"p{i}" for i in range(8)], list(range(8)), top_n=3)
    assert len(rep["ties"]) == 3 and rep["n_ties_priced"] == 28
    assert merge_candidates(np.eye(1), ["only"], [1.0]) is None  # nothing to tie
    assert merge_candidates(np.eye(2), ["a", "b"], [float("nan"), 1.0]) is None
    # a mis-sized filter table is unusable input like any other, not an IndexError out of the tool
    assert merge_candidates(np.eye(3), list("abc"), [1.0, 2.0, 3.0], scales=["log10"] * 2) is None
    assert merge_candidates(np.eye(3), list("abc"), [1.0, 2.0, 3.0], at_bound=[False]) is None
    assert merge_candidates(np.eye(2), ["a", "b"], [1.0, 2.0], n_data=-5)["ties"]  # no ln of -5
    # curving DOWN is not a minimum, so no pair has a constrained-optimum cost to report
    assert merge_candidates(np.diag([4.0, -4.0]), ["a", "b"], [1.0, 2.0]) is None

    # a cost that overflows a double is not a candidate; `Infinity` is not valid strict JSON, and
    # 2*dnll is what goes in dscore — a cost half the double ceiling overflows one line later
    for th in ([1e150, -1e150], [2e104, 0.0]):
        huge = merge_candidates(1e100 * np.eye(2), ["a", "b"], th, n_data=252)
        assert huge["ties"] == [] and huge["n_ties_skipped"] == {"cost_not_finite": 1}
        json.dumps(huge, allow_nan=False)


def test_stats_core_passes_scale_and_at_bound_to_the_merge_report():
    """The filters are only worth anything if the assembly path supplies them: `params_meta`
    carries both at the call site."""
    fim = np.array([[100.0, 99.0], [99.0, 100.0]])
    meta = _meta("a", "b", est_scaled=1.0)
    meta[1]["estimate_scaled"] = 1.02
    assert (
        _identifiability_stats(meta, fim, {}, n_data=252)["merge_candidates"]["n_ties_priced"] == 1
    )

    meta[0]["at_bound"] = True
    pinned = _identifiability_stats(meta, fim, {}, n_data=252)["merge_candidates"]
    assert pinned["ties"] == [] and pinned["n_ties_skipped"] == {"at_bound": 1}

    meta[0]["at_bound"] = False
    meta[0]["scale"] = "lin"
    mixed = _identifiability_stats(meta, fim, {}, n_data=252)["merge_candidates"]
    assert mixed["ties"] == [] and mixed["n_ties_skipped"] == {"scale_mismatch": 1}


def test_roles_are_not_computed_when_there_is_no_fim(session, monkeypatch):
    """merge_candidates is the only consumer of the role table and needs the FIM, so on the
    FIM-less path the whole-model walk would be built and discarded."""
    import petab_mcp.reconcile as reconcile

    calls = []
    monkeypatch.setattr(reconcile, "parameter_roles", lambda *a, **kw: calls.append(1) or {})
    session.fit_record = _record(session, hess=None, sres=None)
    out = identifiability(session)
    assert out["merge_candidates"] is None and calls == []


# -- the merge HINTS: the rows of this diagnosis that ride on every fit result -------------------
def test_merge_hints_are_the_cheapest_ties_bic_would_pay_for():
    """A projection of the diagnosis, never a second statistic — so the hints and the section can
    never name different pairs at the same optimum."""
    from petab_mcp.identifiability import merge_hints

    fim = np.array([[1e4, 1e4 - 1e-5], [1e4 - 1e-5, 1e4]])  # a near-flat difference direction
    meta = _meta("a", "b", est_scaled=1.0)
    meta[1]["estimate_scaled"] = 1.0001
    diag = _identifiability_stats(meta, fim, {}, n_data=252)
    row = diag["merge_candidates"]["ties"][0]
    assert row["merge_improves_score"] is True

    hints = merge_hints(diag)
    assert hints == [{k: row[k] for k in ("parameters", "dnll", "dscore", "move_scaled")}]
    # `move_scaled` rides along because merge_candidates says the price must be read against it — a
    # near-free tie bought with a move of many decades is a flat direction, not an edit. Everything
    # else stays in the section: this is re-sent in history on every later turn.
    assert set(hints[0]) == {"parameters", "dnll", "dscore", "move_scaled"}


def test_merge_hints_report_every_tie_the_charge_covers_up_to_the_cap():
    """One hint per fit taught an agent to merge ONE pair per refit while the rest of the redundancy
    sat unnamed — and the ties come in groups (a saturated model's jointly-redundant
    parameterisation), so the cheapest alone is an arbitrary member of a set. Cheapest-first, and
    capped because this rides on every fit result and is re-sent in history every later turn."""
    import petab_mcp.identifiability as I

    # Five parameters at one shared value with a near-singular FIM: every pair ties for free, so the
    # ranked list is longer than the cap and the cap is what decides the length.
    k = 5
    fim = np.full((k, k), 1e4) + np.eye(k) * 1e-5
    meta = _meta(*"abcde", est_scaled=1.0)
    diag = _identifiability_stats(meta, fim, {}, n_data=252, top_n=10)
    ties = diag["merge_candidates"]["ties"]
    assert sum(1 for r in ties if r["merge_improves_score"]) > I.MERGE_HINT_LIMIT

    hints = I.merge_hints(diag)
    assert len(hints) == I.MERGE_HINT_LIMIT == 3
    # the section's own ranking, truncated — not a re-ranking, and never a row the charge misses
    assert [h["parameters"] for h in hints] == [
        r["parameters"] for r in ties if r["merge_improves_score"]
    ][: I.MERGE_HINT_LIMIT]
    assert all(h["dscore"] < 0 for h in hints)
    assert I.merge_hints(diag, limit=1) == hints[:1]


def test_merge_hints_are_none_when_nothing_beats_the_charge_or_nothing_is_priced():
    """ABSENCE therefore means "no cheap tie OR no FIM OR every pair filtered" — which is why the
    docs must not sell it as a certificate of parsimony."""
    from petab_mcp.identifiability import merge_hints

    # priced, but the tie costs more than the ln(n)/2 the merged-away parameter would save: two
    # well-determined parameters two decades apart
    dear_meta = _meta("a", "b", est_scaled=1.0)
    dear_meta[1]["estimate_scaled"] = 3.0
    dear = _identifiability_stats(dear_meta, np.diag([4.0, 4.0]), {}, n_data=252)
    dear_ties = dear["merge_candidates"]["ties"]
    assert dear_ties and not any(r["merge_improves_score"] for r in dear_ties)
    assert merge_hints(dear) is None  # None, never [] — an omitted key, not an empty list
    # no FIM -> no priced tie at all
    assert merge_hints(_identifiability_stats(_meta("a", "b"), None, {})) is None
    # ...and every unusable shape is None rather than an exception: this is read on the fit path
    assert merge_hints(None) is None and merge_hints({}) is None
    assert (
        merge_hints({"merge_candidates": {"ties": [{"parameters": ["a", "b"], "dnll": 0.0}]}})
        is None
    )  # no price was computable, so no claim that it beats one


def test_fit_merge_hints_never_raises(session, monkeypatch):
    """The fit path calls this; an advisory that can raise is an advisory that can fail a commit."""
    import petab_mcp.identifiability as I

    session.fit_record = _record(session)
    monkeypatch.setattr(
        I, "identifiability", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert I.fit_merge_hints(session) is None


# -- the frozen fit record: what a diagnosis is allowed to read ---------------------------------
def test_only_a_frozen_fit_may_be_diagnosed(session):
    """REGRESSION: the whole module is a local expansion around an optimum — the merge price has no
    gradient term, so off one it gets monotonically CHEAPER (measured: 0.001 nll at the optimum,
    7e-05 at 3x those values) and advertises merges that are not there. With no record there is
    nothing to expand around, and no fallback may reconstruct a problem from live state: that
    reconstruction is what paid a 12.5 s AMICI build to price a model nobody had fitted."""
    import petab_mcp.identifiability as I

    out = identifiability(session)  # nothing fitted in this session
    assert out == {"available": False, "method": "none", "note": I.NO_OPTIMUM_NOTE}
    assert I.fit_merge_hints(session) is None
    assert I.stored_merge_hints(session) is None


def test_the_diagnosed_point_is_the_fits_own_optimum_not_the_live_table(session):
    """REGRESSION (finding 3 of the review): the FIM used to be taken at
    `parameter_df[nominalValue]`, so it was the fit's optimum only while the post-fit write-back
    happened to still be intact. An agent edit to the table must not move the point a diagnosis
    expands around."""
    from petab.v2.C import NOMINAL_VALUE

    from petab_mcp.identifiability import _params_meta

    record = _record(session, x=(0.0, -1.0), hess=np.diag([4.0, 4.0]))
    session.fit_record = record
    before = identifiability(session)["parameters"]

    session.problem.parameter_df.loc["k1", NOMINAL_VALUE] = 42.0  # an edit after the fit
    after = identifiability(session)["parameters"]

    assert after == before
    assert after[0]["estimate"] == pytest.approx(1.0)  # 10**0.0, the FITTED value, not 42.0
    # ...and the point the quadratic is expanded around is the record's, coordinate for coordinate
    assert [p["estimate_scaled"] for p in _params_meta(record)] == [0.0, -1.0]
    # the objective was asked for sensitivities AT that point, not at the edited table's
    assert record.pypesto_problem.objective.calls[0][0] == (0.0, -1.0)


def test_another_structure_in_the_working_state_is_labelled_not_re_priced(session):
    """A restore points the working state at another version's model, so the geometry in hand
    describes a structure that is no longer in front of the agent. The diagnosis then belongs to the
    LAST FIT and says so — re-pricing the current structure would mean building it."""
    import petab_mcp.identifiability as I

    session.fit_record = _record(
        session, hess=np.diag([4.0, 4.0]), structure_hash="another-structure"
    )
    out = identifiability(session)

    assert out["stale"] is True and out["stale_note"] == I.STALE_NOTE
    assert out["available"] is True  # still the honest numbers of that earlier fit
    # ...and nothing rides on the fit RESULT or the score read off another structure's state
    session.simulation = {
        I.STORED_DIAGNOSIS_KEY: {"structure_hash": "another-structure", "diagnosis": out}
    }
    assert I.stored_merge_hints(session) is None
    assert I.fit_merge_hints(session) is None


def test_a_table_only_drift_keeps_the_verdict_and_says_the_table_moved(session):
    """REGRESSION: staleness used to compare `problem_hash` (the model AND the parameter table), so
    the ONE drift an agent can cause inside a session — the de-dup no-op, which re-authors the SAME
    structure with a new starting guess and therefore runs no fit — read as "this describes another
    model". That dropped the merge hint from the score read and switched OFF the parsimony finish
    gate for a rank deficiency the frozen FIM still reported, for the same structure, at HEAD's own
    fit. The structure is what decides; the moved values are reported."""
    import petab_mcp.identifiability as I

    session.fit_record = _record(session, x=(1.0, 1.0001), hess=_NEAR_FLAT)
    undrifted = identifiability(session)
    session.fit_record = _record(
        session, x=(1.0, 1.0001), hess=_NEAR_FLAT, problem_hash="the-fit-before-the-no-op"
    )

    out = identifiability(session)
    assert {k: v for k, v in out.items() if k != "working_table_note"} == undrifted
    assert "stale" not in out
    assert out["working_table_note"] == I.WORKING_TABLE_NOTE
    # ...and the score read still names the tie the fit priced, since HEAD is still that fit
    session.simulation = {
        I.STORED_DIAGNOSIS_KEY: {
            "structure_hash": session.fit_record.structure_hash,
            "diagnosis": out,
        }
    }
    assert [h["parameters"] for h in I.stored_merge_hints(session)] == [["k1", "k2"]]


def test_a_process_that_ran_no_fit_serves_the_verdict_the_fit_stored(session):
    """REGRESSION: a durable RESUME restores HEAD's payload but not the in-memory fit record, so the
    parameter half went dark for the rest of the session — and a finish gate stopped refusing a
    rank-deficient model it refuses in-process. Re-deriving a record there is what the invariant
    forbids: measured on the pre-fix code, a resumed diagnosis paid 19.4 s of AMICI codegen (122 new
    files under the build root, charged to no clock), because a restored model's SBML is serialised
    in a different order and hashes to a module nothing has built."""
    import petab_mcp.identifiability as I

    fitted = identifiability(_fitted(session, x=(1.0, 1.0001), hess=_NEAR_FLAT))
    resumed = _resumed(session, fitted)

    out = identifiability(resumed)
    assert out["available"] is True and "stale" not in out
    assert out["fim"] == fitted["fim"] and out["parameters"] == fitted["parameters"]
    assert [h["parameters"] for h in I.stored_merge_hints(resumed)] == [["k1", "k2"]]
    # ...and a resume onto a DIFFERENT structure is labelled, not re-priced
    resumed.simulation[I.STORED_DIAGNOSIS_KEY]["structure_hash"] = "another-structure"
    assert identifiability(resumed)["stale"] is True
    assert I.stored_merge_hints(resumed) is None


#: A FIM whose DIFFERENCE direction is nearly flat: the state a merge hint exists in (the pair is
#: free to tie) while every standard error stays finite, so a payload built on it is fully populated.
_NEAR_FLAT = np.array([[1e4, 1e4 - 1e-5], [1e4 - 1e-5, 1e4]])


def _fitted(session, **kw):
    """``session`` with a frozen record, as the process that ran the fit holds it."""
    session.fit_record = _record(session, **kw)
    return session


def _resumed(session, diagnosis):
    """A session in the state a RESUMED process is in: HEAD's payload restored (the stored verdict
    with it), no fit record, no engine — the fits happened before this process started."""
    import copy

    from petab_mcp.identifiability import STORED_DIAGNOSIS_KEY

    resumed = copy.copy(session)
    resumed.fit_record = None
    resumed.simulation = {
        "status": "fit",
        STORED_DIAGNOSIS_KEY: {
            "structure_hash": session.fit_record.structure_hash,
            "diagnosis": copy.deepcopy(diagnosis),
        },
    }
    return resumed


def test_a_hierarchical_fits_rank_is_read_against_its_own_dimension(session):
    """The fit's objective inner-solves the sigmas, so the FIM is REDUCED: it has no sigma
    coordinate even though the session counts one in `k`. Every rank statement must therefore be
    made against the FIM's OWN dimension — comparing it with the session's k would read every
    hierarchical fit as rank-deficient and the finish gate would refuse it."""
    from petab.v2.C import ESTIMATE

    df = session.problem.parameter_df
    df.loc["sigma_B", ESTIMATE] = True  # k = 3 for the session...
    assert int(session.problem.n_estimated) == 3
    # ...while the fit's outer problem carries only k1, k2 (sigma_B was inner-solved)
    session.fit_record = _record(session, hess=np.diag([4.0, 4.0]), hierarchical=True)

    out = identifiability(session)

    assert out["n_estimated"] == 2 and out["fim"]["n_estimated"] == 2
    assert out["fim"]["rank"] == 2 and out["fim"]["rank_deficient"] is False
    assert [p["id"] for p in out["parameters"]] == ["k1", "k2"]
    # the omission is stated, so it does not read as a bug
    assert "inner-solved" in out["inner_parameters_omitted"]
