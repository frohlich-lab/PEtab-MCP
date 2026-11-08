"""Parameter-identifiability diagnosis of the current fit.

Which estimated parameters are superfluous? An agent over-parameterises because raw nll/score
don't say WHICH parameters are unconstrained or trading off. The ``parameters`` section of
``get_fit_diagnosis`` (see :mod:`petab_mcp.server`) answers that from two cheap, complementary
signals at the fitted optimum:

* the Fisher Information Matrix (FIM ≈ Hessian of the nllh, from AMICI sensitivities) ->
  per-parameter standard error, marginal + partial correlations, and — via its
  eigendecomposition — the non-identifiable parameter COMBINATIONS (flat directions);
* the multistart spread: how much each parameter wanders across near-optimal starts (a
  model-free practical-identifiability check that needs no Hessian).

The FIM is a LOCAL linearization and can understate non-identifiability for strongly nonlinear
models, so the two are meant to be read together. All output is bounded (top-N pairs/directions).

One verdict off that diagnosis rides on EVERY fit result instead of waiting to be asked for:
:func:`fit_merge_hints` names the parameter ties whose cost the complexity penalty already pays for (see
:func:`merge_hints`). It is the same computation the section reports, condensed — a tool called
once a session cannot inform a decision made every turn.

NOTHING HERE MAY BUILD A MODEL. A diagnosis describes a fit that already happened, and the module
that fit was compiled with already exists, so every entry point reads the frozen
:class:`fitting.FitRecord` the fit left behind — its pyPESTO problem, its point, its tables — and
never re-derives an AMICI output folder from live session state. That is the whole reason the
record exists; see its docstring for the build a live read used to pay for. A session outlives its
process, so where there is no record the verdict the fit STORED is re-served instead
(:func:`fit_diagnosis_block`) — again computing nothing. The working model having moved since is
reported (``stale``), never re-priced. ``tests/test_diagnosis_never_compiles.py`` holds that line.

Split out of :mod:`petab_mcp.fitting` (which owns fit/simulate orchestration) so this
analytical subsystem stands on its own.
"""

from __future__ import annotations

import math
from typing import Any

from . import criteria
from .constants import PARAMETER_AT_BOUND_RELATIVE_TOL


def _identifiability_stats(
    params_meta: list[dict[str, Any]],
    fim,
    spread: dict[str, Any],
    *,
    top_n: int = 8,
    corr_threshold: float = 0.9,
    flat_rel: float = 1e-3,
    se_unconstrained: float = 1.0,
    spread_unstable: float = 0.5,
    roles: dict[str, str] | None = None,
    fixed_tables: dict[str, str] | None = None,
    n_data: int | None = None,
    criterion: "criteria.Criterion | None" = None,
) -> dict[str, Any]:
    """Pure statistics core (no pyPESTO/AMICI). ``params_meta`` is one dict per estimated
    parameter — ``{id, scale, estimate, estimate_scaled, at_bound}`` — in the SAME order as the
    rows/cols of ``fim`` (an ``m×m`` numpy array or ``None``). ``spread`` maps parameter id ->
    ``{std,min,max,n}`` from the multistart. Returns the compact, bounded diagnosis dict."""
    import numpy as np

    names = [p["id"] for p in params_meta]
    m = len(names)
    per = {
        p["id"]: {
            "id": p["id"],
            "scale": p.get("scale"),
            "estimate": p.get("estimate"),
            "at_bound": bool(p.get("at_bound")),
            "se_scaled": None,
            "fold_uncertainty": None,
            "multistart_std": None,
            "multistart_range": None,
        }
        for p in params_meta
    }

    # multistart spread (always available if the payload carried per-start x)
    for pid, s in (spread or {}).items():
        if pid in per and isinstance(s, dict):
            per[pid]["multistart_std"] = s.get("std")
            if s.get("min") is not None and s.get("max") is not None:
                per[pid]["multistart_range"] = round(float(s["max"]) - float(s["min"]), 4)

    correlations: list[dict[str, Any]] = []
    partials: list[dict[str, Any]] = []
    flat_directions: list[dict[str, Any]] = []
    fim_info = None

    if fim is not None and m >= 1:
        H = np.asarray(fim, dtype=float)
        cov = np.linalg.pinv(H)
        var = np.clip(np.diag(cov), 0.0, None)
        se = np.sqrt(var)
        for i, pid in enumerate(names):
            per[pid]["se_scaled"] = round(float(se[i]), 4)
            sc = (params_meta[i].get("scale") or "").lower()
            if np.isfinite(se[i]):
                # a legible multiplicative uncertainty for log-scaled params
                if sc in ("log10", "log"):
                    base = 10.0 if sc == "log10" else math.e
                    # base ** se overflows to inf for a practically-unidentifiable parameter
                    # (huge-but-finite se) — keep the JSON NaN/Inf-free like the rest of the API.
                    fold = float(base ** se[i])
                    if math.isfinite(fold):
                        per[pid]["fold_uncertainty"] = round(fold, 3)
        # marginal correlations from the covariance
        dvar = np.sqrt(np.outer(var, var))
        with np.errstate(divide="ignore", invalid="ignore"):
            marg = np.where(dvar > 0, cov / dvar, 0.0)
        # partial correlations from the FIM (precision matrix) itself
        dh = np.sqrt(np.outer(np.abs(np.diag(H)), np.abs(np.diag(H))))
        with np.errstate(divide="ignore", invalid="ignore"):
            pcorr = np.where(dh > 0, -H / dh, 0.0)
        for i in range(m):
            for j in range(i + 1, m):
                if abs(marg[i, j]) >= corr_threshold:
                    correlations.append(
                        {"a": names[i], "b": names[j], "r": round(float(marg[i, j]), 3)}
                    )
                if abs(pcorr[i, j]) >= corr_threshold:
                    partials.append(
                        {"a": names[i], "b": names[j], "pr": round(float(pcorr[i, j]), 3)}
                    )
        correlations.sort(key=lambda d: -abs(d["r"]))
        partials.sort(key=lambda d: -abs(d["pr"]))
        correlations = correlations[:top_n]
        partials = partials[:top_n]
        # flat directions: eigenvectors with near-zero eigenvalue = non-identifiable combinations
        evals, evecs = np.linalg.eigh(H)
        emax = float(evals.max()) if evals.size else 0.0
        for idx in range(len(evals)):
            ev = float(evals[idx])
            rel = ev / emax if emax > 0 else 0.0
            if emax > 0 and rel < flat_rel:
                vec = evecs[:, idx]
                order = np.argsort(-np.abs(vec))
                participants = [
                    {"id": names[k], "weight": round(float(vec[k]), 3)}
                    for k in order[:4]
                    if abs(vec[k]) > 0.15
                ]
                flat_directions.append(
                    {"eigenvalue": round(ev, 6), "rel": round(rel, 6), "participants": participants}
                )
        rank = int(np.linalg.matrix_rank(H))
        cond = float(
            emax / max(float(evals[evals > 0].min()) if (evals > 0).any() else emax, 1e-300)
        )
        fim_info = {
            "rank": rank,
            "n_estimated": m,
            "condition": round(cond, 1),
            "rank_deficient": rank < m,
        }

    # simplification candidates: rank estimated params by identifiability red flags
    corr_partner: dict[str, list] = {}
    for c in correlations:
        corr_partner.setdefault(c["a"], []).append((c["b"], c["r"]))
        corr_partner.setdefault(c["b"], []).append((c["a"], c["r"]))
    flat_members = {p["id"] for d in flat_directions for p in d["participants"]}
    candidates = []
    for pid in names:
        p = per[pid]
        reasons = []
        if p["at_bound"]:
            reasons.append("at_bound")
        if p["se_scaled"] is not None and p["se_scaled"] >= se_unconstrained:
            reasons.append(f"poorly_constrained(se={p['se_scaled']})")
        if pid in corr_partner:
            b, r = max(corr_partner[pid], key=lambda t: abs(t[1]))
            reasons.append(f"collinear_with:{b}(r={r})")
        if pid in flat_members:
            reasons.append("flat_direction")
        if p["multistart_std"] is not None and p["multistart_std"] >= spread_unstable:
            reasons.append(f"multistart_unstable(std={round(p['multistart_std'], 3)})")
        if reasons:
            candidates.append({"id": pid, "n_flags": len(reasons), "reasons": reasons})
    candidates.sort(key=lambda d: -d["n_flags"])

    # Which pairs could take ONE shared value — the move flat_directions cannot express (it names
    # the members of a poorly determined combination, not which TIE is cheap). Same FIM, no extra
    # simulation. scale and at_bound go with it: both decide whether a pair can be priced honestly.
    merges = None
    if fim is not None:
        theta = [p.get("estimate_scaled") for p in params_meta]
        if all(t is not None for t in theta):
            merges = merge_candidates(
                fim,
                names,
                theta,
                scales=[p.get("scale") for p in params_meta],
                at_bound=[per[n]["at_bound"] for n in names],
                roles=roles,
                fixed_tables=fixed_tables,
                n_data=n_data,
                top_n=top_n,
                criterion=criterion,
            )

    method = (
        "fim+multistart"
        if fim is not None and spread
        else "fim"
        if fim is not None
        else "multistart"
        if spread
        else "none"
    )
    return {
        "available": method != "none",
        "method": method,
        "n_estimated": m,
        "fim": fim_info,
        "parameters": [per[n] for n in names],
        "correlations": correlations,
        "partial_correlations": partials,
        "flat_directions": flat_directions[:top_n],
        "candidates": candidates[:12],
        "merge_candidates": merges,
        "notes": (
            "FIM is a LOCAL linearization of uncertainty — read se/correlations together "
            "with multistart_std (a model-free spread) and flat_directions (the actual "
            "non-identifiable parameter combinations). High |r|≈1, large fold_uncertainty, "
            "at_bound, or a flat direction all mark a parameter as a simplification "
            "candidate: remove its term, take its limiting (irreversible/instantaneous) "
            "form, or merge a collinear pair into one parameter — all MODEL edits. "
            "`merge_candidates` prices the ONE-SHARED-VALUE form of that merge, cheapest "
            "first, with the parameter move each tie would take. "
            "If none fits cleanly, leave it: there is no value/estimate "
            "tool, and a hard-coded literal is a magic number; values come only from the fit."
        ),
    }


def _compute_fim(record):
    """FIM (≈ Hessian of the nllh) at the FROZEN fitted point, in the record's free-scaled space.

    Evaluated on ``record.pypesto_problem`` — the very objective the optimizer minimised, whose
    AMICI module is already imported — so this costs ONE sensitivity evaluation and cannot build
    anything. There is no importer here, and no hash to resolve: that is the invariant, not an
    optimisation (see :class:`fitting.FitRecord`).

    The matrix is :func:`_fim_from_rows`' — the same construction, off the same evaluation, that
    :func:`effective_parameter_count` takes its ``H`` from, so the FIM the agent-facing statistics
    invert and the one the effective count divides by are one object. The objective's own ``hess``
    and ``sres.T @ sres`` remain as fallbacks, in that order, for an objective with no residual
    mode; ``docs/RATIONALE.md`` has why neither leads.

    On a HIERARCHICAL fit that objective inner-solves the noise sigmas / observable scalings, so
    the rows carry only the OUTER coordinates and this is the REDUCED FIM. That is the right
    object: the inner parameters are shipped by the observation model, so they define no mergeable
    direction, and every rank statement is made against the FIM's OWN dimension (see
    :func:`_identifiability_stats`) — never against the session's ``k``, which counts them.

    Returns the ``m×m`` array or ``None`` (best-effort — a missing FIM just drops the correlation
    half of the diagnosis, leaving the multistart spread)."""
    try:
        import numpy as np

        problem = record.pypesto_problem
        x_free = np.asarray(record.x_free, dtype=float)
        mfree = len(record.x_names)

        def _finite_square(a):
            a = np.asarray(a, dtype=float)
            return a.shape == (mfree, mfree) and np.isfinite(a).all()

        try:
            built = _fim_from_rows(problem, x_free)
            if built is not None and _finite_square(built[0]):
                return np.asarray(built[0], dtype=float)
        except Exception:  # noqa: BLE001 - no residual mode here; ask for the Hessian instead
            pass
        try:
            ret = problem.objective(x_free, sensi_orders=(0, 1, 2), return_dict=True)
            for key in ("hess", "HESS"):
                if key in ret and _finite_square(ret[key]):
                    return np.asarray(ret[key], dtype=float)
        except Exception:  # noqa: BLE001 - order-2 unsupported for this objective; try sres
            pass
        try:
            ret = problem.objective(x_free, sensi_orders=(0, 1), return_dict=True)
            for key in ("sres", "SRES"):
                if key in ret and ret[key] is not None:
                    sres = np.asarray(ret[key], dtype=float)
                    fim = sres.T @ sres
                    if _finite_square(fim):
                        return fim
        except Exception:  # noqa: BLE001
            pass
        return None
    except Exception:  # noqa: BLE001 - unevaluable objective; degrade to spread-only
        return None


def _residual_rows(problem, x):
    """``(res, sres, blocks)`` of the built ``problem`` at the FREE point ``x``, in residual mode:
    the residual vector, its sensitivities with respect to ``x`` (one column per free parameter),
    and the per-experiment row counts pyPESTO reports through ``rdatas``.

    A plain objective answers this itself (AMICI's ``sres`` is on the estimation scale already,
    see :func:`fitting.pypesto_problem`). A HIERARCHICAL one refuses: pyPESTO raises on residual sensitivities whenever a
    sigma is inner-solved, since at the analytic sigma the least-squares objective is constant —
    true for an optimizer, irrelevant to the FIM, which only needs the rows at one point. So the
    inner problem is solved once at ``x`` (a function-mode call, which reports the inner values),
    and the rows are evaluated by pyPESTO's PLAIN v2 calculator on the hierarchical objective's own
    simulator with those values filled in: the same module, the same solver, nothing built. Its
    ``sres`` spans the outer parameters followed by the inner ones; the inner columns are dropped
    (the FIM has no coordinate for them, see :data:`INNER_OMITTED_NOTE`) and the outer ones, already
    on the estimation scale, are cut to ``x_free_indices``. Without the detour the count is ``None``
    for every fit with an inner-solved sigma, i.e. most real examples; scores built from these rows
    agree with the profiled objective's own gradient, which is the envelope theorem.
    """
    import numpy as np
    from pypesto.C import INNER_PARAMETERS, MODE_RES, RDATAS, RES, SRES
    from pypesto.objective.amici.amici_calculator import AmiciCalculatorPetabV2

    obj = problem.objective
    calc = getattr(obj, "calculator", None)
    if hasattr(calc, "inner_calculators"):
        inner_ids = list(calc.get_interpretable_inner_par_ids())
        inner_values = obj(x, sensi_orders=(0,), return_dict=True)[INNER_PARAMETERS]
        x_full = obj.pre_post_processor.preprocess(x=x)
        outer_ids = list(obj.x_ids)
        ret = AmiciCalculatorPetabV2(obj._petab_simulator)(
            x_dct={**dict(zip(outer_ids, x_full)), **dict(zip(inner_ids, inner_values))},
            sensi_orders=(0, 1),
            mode=MODE_RES,
            amici_model=obj.amici_model,
            amici_solver=obj.amici_solver,
            edatas=obj.edatas,
            n_threads=obj.n_threads,
            x_ids=outer_ids + inner_ids,
            parameter_mapping=None,
            fim_for_hess=obj.fim_for_hess,
        )
        sres = np.asarray(ret[SRES], dtype=float)[:, list(problem.x_free_indices)]
    else:
        ret = obj(x, sensi_orders=(0, 1), mode=MODE_RES, return_dict=True)
        sres = np.asarray(ret[SRES], dtype=float)
    res = np.asarray(ret[RES], dtype=float).reshape(-1)
    return res, sres, [len(r["res"]) for r in ret[RDATAS]]


def _fim_from_rows(problem, x):
    """``(H, S)`` at the free point ``x`` of the built pyPESTO ``problem``: the Fisher information,
    and the per-measurement score rows it is summed from. ``None`` where the rows are not the
    layout this pairing assumes, or the objective's residual mode cannot be evaluated there.

    Both come off ONE evaluation of that objective's RESIDUAL mode with AMICI's sigma
    pseudo-residuals on (:func:`_residual_rows`): each measurement then contributes two rows —
    ``r = (m−y)/σ`` and ``ρ = √(2·ln σ + c)`` — whose ``res·sres`` products sum EXACTLY to that
    point's score, the ``−∂ln σ`` term an estimated sigma adds included (the constant ``c`` cancels
    in the product). AMICI lays an experiment's rows out as the data half followed by the sigma
    half, index-aligned, which is how they are paired here, and the flag is set for this one
    evaluation and restored.

    ``H`` is the Gauss-Newton information over the same rows, the data rows as they are and the
    sigma rows weighted by ``ρ²`` — the exact Gaussian Fisher information
    ``∂y∂yᵀ/σ² + 2·∂ln σ∂ln σᵀ`` per point, and NOT the ``hess`` the objective reports, which
    leaves an ESTIMATED sigma at half its Fisher information. Both consumers invert it — the
    agent-facing statistics of :func:`_identifiability_stats` through :func:`_compute_fim`, and
    :func:`effective_parameter_count` — so they take it from here rather than each building their
    own; ``docs/RATIONALE.md`` has the measured difference and why the flag is toggled by hand.

    Raises whatever the objective raises: the callers differ in what they do with an objective that
    has no residual mode (a fallback FIM, or no count at all), so that decision is theirs.
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    m = int(x.size)
    if m == 0 or not np.isfinite(x).all():
        return None

    model = problem.objective.amici_model
    before = model.get_add_sigma_residuals()
    model.set_add_sigma_residuals(True)
    try:
        res, sres, blocks = _residual_rows(problem, x)
    finally:
        model.set_add_sigma_residuals(before)
    if sres.shape != (res.size, m) or not (np.isfinite(res).all() and np.isfinite(sres).all()):
        return None
    if not blocks or sum(blocks) != res.size or any(b % 2 for b in blocks):
        return None
    rows = res[:, None] * sres  # one row per residual: its res·∂res term
    scores, start = [], 0
    H = np.zeros((m, m))
    for b in blocks:  # fold each experiment's sigma half onto its data half
        half = b // 2
        data, sigma = slice(start, start + half), slice(start + half, start + b)
        scores.append(rows[data] + rows[sigma])
        # ∂r∂rᵀ from the data rows, ρ²·∂ρ∂ρᵀ = ∂ln σ∂ln σᵀ from the sigma rows.
        H += sres[data].T @ sres[data] + rows[sigma].T @ rows[sigma]
        start += b
    return H, np.vstack(scores)


def effective_parameter_count(problem, x_free, *, k: int | None = None) -> float | None:
    """TIC's effective parameter count ``k_eff = tr(J·H⁻¹)`` at the FROZEN optimum ``x_free`` of
    the built pyPESTO ``problem`` — or ``None`` when it cannot be evaluated there.

    How many parameters the DATA see, as opposed to how many the model declares (``k``):
    Takeuchi's trace, with ``J = Σ_i sᵢsᵢᵀ`` over the data points' score vectors (each point's
    gradient of its own log-likelihood term) and ``H`` the Fisher information at the optimum, so
    ``k_eff = Σ_i sᵢᵀH⁻¹sᵢ`` — one leverage-weighted term per measurement, no integral over
    parameter space. It equals ``k`` for a correctly specified model (there ``E[ssᵀ] = H``),
    exceeds it where large residuals sit on high-leverage points, and falls short of it along
    directions the data do not determine — those are dropped from ``H⁻¹`` at numpy's rank
    tolerance, the floor every FIM inversion in this module uses — so ``k_eff`` well below ``k``
    says the model carries parameters the data cannot tell apart. It is also what TIC charges for
    in place of ``k``.

    Both ``J`` and ``H`` come off ONE evaluation, through :func:`_fim_from_rows` — the same call
    that gives the diagnosis its FIM, so the matrix inverted here and the one the agent reads
    standard errors and merge prices off cannot disagree.

    ``k`` is the count the score charges for, :func:`fitting.complexity_k`; a hierarchical fit's
    inner-solved parameters (sigmas, scalings, offsets) and the model's hardcoded constants have no
    coordinate in ``x_free`` and no row in ``H``, and are charged one unit each (the ``k − m`` of
    them) rather than dropped, because a profiled parameter the data pin is exactly where the count
    and the trace agree, and a literal left out would be free under TIC alone. Best-effort in the
    strong sense: ANY failure is ``None``, and ``None`` is "no count", never a fallback to ``k`` —
    a ``k`` standing in for a missing measurement would read as "the data determine every
    parameter", the one conclusion this statistic exists to be able to contradict. Evaluates the
    built objective and nothing else — no importer, no hash, so no build (see
    :class:`fitting.FitRecord`). One sensitivity evaluation, the FIM's cost.

    ``docs/RATIONALE.md`` has why ``H`` is built from the rows rather than taken from the
    objective, and why the count reaches no agent-facing result.
    """
    try:
        import numpy as np

        built = _fim_from_rows(problem, x_free)
        if built is None:
            return None
        H, S = built
        m = int(H.shape[0])

        evals, evecs = np.linalg.eigh(H)
        eps = float(np.finfo(float).eps)
        tol = max(m * eps * float(np.abs(evals).max(initial=0.0)), float(np.finfo(float).tiny))
        keep = evals > tol
        if not keep.any():
            return None
        Hinv = (evecs[:, keep] / evals[keep]) @ evecs[:, keep].T
        k_eff = float(np.einsum("ij,jk,ik->", S, Hinv, S))
        if k is not None and int(k) > m:
            k_eff += int(k) - m
        return k_eff if math.isfinite(k_eff) and k_eff >= 0.0 else None
    except Exception:  # noqa: BLE001 - advisory statistic; a fit is never failed by it
        return None


def _multistart_spread(optimization, free_names: list[str]) -> dict[str, Any]:
    """Per-parameter spread across near-optimal multistarts (scaled space), from the FROZEN
    optimization payload's per-start ``x``. Empty when x/x_names were not recorded."""
    import numpy as np

    opt = optimization or {}
    starts = opt.get("starts") or []
    xnames = opt.get("x_names") or []
    if not starts or not xnames:
        return {}
    col = {n: i for i, n in enumerate(xnames)}
    fvals = [
        float(s["fval"])
        for s in starts
        if s.get("fval") is not None and math.isfinite(float(s["fval"]))
    ]
    if not fvals:
        return {}
    best = min(fvals)
    # near-optimal band; fall back to the best few starts if the band is too tight
    band = [
        s
        for s in starts
        if s.get("fval") is not None
        and s.get("x") is not None
        and math.isfinite(float(s["fval"]))
        and float(s["fval"]) <= best + 1.0
    ]
    if len(band) < 3:
        band = sorted(
            [s for s in starts if s.get("x") is not None and s.get("fval") is not None],
            key=lambda s: float(s["fval"]),
        )[:5]
    out: dict[str, Any] = {}
    for n in free_names:
        j = col.get(n)
        if j is None:
            continue
        vals = [float(s["x"][j]) for s in band if s.get("x") is not None and j < len(s["x"])]
        if len(vals) >= 2:
            arr = np.asarray(vals)
            out[n] = {
                "std": float(arr.std()),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "n": len(vals),
            }
    return out


def _measurement_count(problem) -> int | None:
    """Number of measurement rows — the ``n`` in the complexity penalty, so a tie's cost can be
    compared against what the criterion charges for the parameter it would remove."""
    try:
        n = len(problem.measurement_df)
        return n or None
    except Exception:  # noqa: BLE001 - no table: report the ties without a price
        return None


def _scale_key(s) -> str:
    """A parameterScale normalised for comparison; ``""`` for anything unusable (NaN, missing)."""
    return s.strip().lower() if isinstance(s, str) else ""


def _cost_nll(x: float) -> float:
    """A cost as DISPLAYED. Three decimals reads against the ln(n)/2 charge, but rounding a positive
    cost to 0.0 would print a direction the data never determined and a 4e-4 tie as the same row;
    below 1e-9 nll there is nothing left to tell apart."""
    r = round(x, 3)
    return r if r else (float(f"{x:.1g}") if x > 1e-9 else 0.0)


def merge_candidates(
    fim,
    names,
    theta,
    *,
    scales=None,
    at_bound=None,
    roles=None,
    fixed_tables=None,
    n_data=None,
    top_n=6,
    criterion=None,
):
    """Which pairs of estimated parameters could take ONE SHARED VALUE, and what that would cost.

    The question ``flat_directions`` does not answer: it names the members of a poorly determined
    combination, not which TIE is cheap, and those are different projections of the same matrix.

    What is priced is exactly ``theta_i == theta_j`` on the estimation scale — the edit that deletes
    one parameter by making the two the same symbol. It is not a substitution that replaces the pair
    with one derived quantity: that constrains a different direction and costs something else.
    Under the local quadratic the rest of this module assumes, imposing ``c @ theta = 0`` on the
    optimum costs

        dnll = (c @ theta)**2 / (2 * c @ inv(H) @ c)

    and the shared-value merge is ``c = e_i - e_j``.

    ``inv(H)`` runs through the eigendecomposition with the eigenvalues CLAMPED at numpy's rank
    tolerance, not truncated by a pseudo-inverse, because truncation inverts the limit on the case
    that matters most. Where H is flat along ``c`` the data never determined that direction:
    ``c @ inv(H) @ c`` diverges and the tie is FREE. A pinv sets ``1/lambda`` to 0 there instead of
    infinity, which collapses the denominator to a rounding artifact and prices the one free merge
    as the most expensive in the report. Clamping keeps the price continuous in H and in the pair's
    overlap with the flat direction, so no threshold on an eigenvector component decides a verdict.

    ``move_scaled`` is how far the furthest parameter has to travel, on the estimation scale, to
    reach that constrained optimum. Read it against the gap the tie closes: a near-free price bought
    with a move of many decades is the quadratic running off down a flat direction rather than an
    edit that will survive a refit.

    A pair is offered only when the merge is an edit someone could write and this quadratic can
    price it honestly — an edit can tie them (``fixed_tables``, from
    :func:`reconcile.fixed_table_parameters`: a member the model's math does not read, a noise
    sigma the fit solves analytically among them, is in no model text to rename, and of two the
    fixed tables read as well, either renamed away leaves them reading its old name; one of those
    is listed FIRST, as the member a tie keeps), same ROLE (nothing stops the
    arithmetic tying a rate to a noise sigma), same parameterScale (``log10(k_a) == k_b`` is not an
    equality anyone means), and neither member AT A BOUND (the expansion assumes an interior
    stationary point, so a pinned parameter's cost would be understated). Each filter needs its
    table: whichever the caller leaves out is named in ``filters_off``, and every pair a filter
    refused is counted by reason in ``n_ties_skipped``, so that "no free tie exists" and "the free
    tie was suppressed" read differently. The ranked list is truncated to ``top_n``.

    Read as a SCREEN. The quadratic is local: it separates cheap ties from expensive ones, but a
    single price can be off by a factor in EITHER direction, so the criterion adjudicates the
    refit that follows a merge.
    """
    import numpy as np

    from .reconcile import READ_BY_TABLES_TOO

    # Resolved ONCE: the price below and the label reported beside it must be the same criterion.
    # The session's, when a session is scoring (see `identifiability`); plain BIC for a bare call.
    crit = criterion or criteria.Criterion()

    H = np.asarray(fim, dtype=float)
    th = np.asarray(theta, dtype=float)
    k = len(names)
    if k < 2 or H.shape != (k, k) or not np.isfinite(H).all() or not np.isfinite(th).all():
        return None
    if any(t is not None and len(t) != k for t in (scales, at_bound)):
        return None  # module-public: a mis-sized table is None like every other unusable input
    evals, evecs = np.linalg.eigh(H)  # the FIM is symmetric by construction
    eps = float(np.finfo(float).eps)
    emax = float(np.abs(evals).max(initial=0.0))
    # numpy's own rank tolerance — the scale at which an eigenvalue stops being distinguishable from
    # zero. Floored at `tiny` so a FIM with no curvature at all still inverts to "nothing is
    # determined" instead of dividing by zero.
    tol = max(k * eps * emax, float(np.finfo(float).tiny))
    if evals.min(initial=0.0) < -math.sqrt(eps) * emax:
        # Curving DOWN by more than the sensitivity solve's own accuracy: the point is not a
        # minimum, so no pair has a constrained-optimum cost. Refuse rather than price it.
        return None
    Hinv = (evecs / np.maximum(evals, tol)) @ evecs.T
    d = np.diag(Hinv)
    curv = d[:, None] + d[None, :] - 2.0 * Hinv  # c @ inv(H) @ c for every c = e_i - e_j at once
    gap = th[:, None] - th[None, :]  # c @ theta likewise

    # What the session's criterion charges for the k-th parameter, in nll (see
    # :meth:`criteria.Criterion.parameter_price`): ln(n)/2 under BIC, 1 under AIC, k-dependent
    # under AICc. A merge that costs less than this pays for itself by deleting a parameter.
    _price = crit.parameter_price(k, n_data)
    price = round(_price, 3) if _price is not None else None
    ties: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}

    def _skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    for i in range(k):
        for j in range(i + 1, k):
            # Each filter FAILS CLOSED, and says which way it closed: "these two differ" is a claim
            # about the pair, "this one is unknown" is a claim about the table, and a reader acts on
            # them differently.
            fi = fj = None
            if fixed_tables is not None:
                fi, fj = fixed_tables.get(names[i]), fixed_tables.get(names[j])
                why = next((f for f in (fi, fj) if f and f != READ_BY_TABLES_TOO), None)
                if why or (fi and fj):
                    _skip(why or READ_BY_TABLES_TOO)
                    continue
            if roles is not None:
                ri, rj = roles.get(names[i]), roles.get(names[j])
                if not (ri and rj):
                    _skip("role_unknown")
                    continue
                if ri != rj:
                    _skip("role_mismatch")
                    continue
            if scales is not None:
                si, sj = _scale_key(scales[i]), _scale_key(scales[j])
                if not (si and sj):
                    _skip("scale_unknown")
                    continue
                if si != sj:
                    _skip("scale_mismatch")
                    continue
            if at_bound is not None and (at_bound[i] or at_bound[j]):
                _skip("at_bound")
                continue
            with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                exact = float(gap[i, j] ** 2 / (2.0 * curv[i, j]))
                # the constrained optimum is -(c@theta) inv(H) c / (c@inv(H)@c); report its
                # largest component, the move an agent would have to believe.
                move = float(abs(gap[i, j]) * np.abs(Hinv[:, i] - Hinv[:, j]).max() / curv[i, j])
            # 2*dnll is the score charge, so a cost within a factor of two of the double ceiling is
            # `Infinity` in the payload — not valid strict JSON, and not a merge candidate either.
            if not (math.isfinite(2.0 * exact) and math.isfinite(move)):
                _skip("cost_not_finite")
                continue
            dnll = _cost_nll(exact)
            pair = [names[j], names[i]] if fj else [names[i], names[j]]
            row = {"parameters": pair, "dnll": dnll, "move_scaled": round(move, 3)}
            if price is not None:
                # Both read off the DISPLAYED cost and price, so the row agrees with itself:
                # dscore < 0 exactly when merge_improves_score is true.
                row["dscore"] = round(2.0 * (dnll - price), 3)
                row["merge_improves_score"] = dnll < price
            ties.append(row)
    # Cheapest first, and among the ties the price cannot separate — the ones the data left
    # undetermined, all of them free — the smallest move first, since that is what decides which of
    # them is a writable edit. Both keys are on the row, so the order can be checked by reading it.
    ties.sort(key=lambda r: (r["dnll"], r["move_scaled"]))
    out: dict[str, Any] = {"ties": ties[:top_n], "n_ties_priced": len(ties)}
    if skipped:
        out["n_ties_skipped"] = dict(sorted(skipped.items()))
    off = [
        n
        for n, table in (
            ("fixed_tables", fixed_tables),
            ("role", roles),
            ("scale", scales),
            ("at_bound", at_bound),
        )
        if table is None
    ]
    if off:  # nothing refused these pairs, so merges nobody could write are in the list too
        out["filters_off"] = off
    if price is not None:
        out["criterion"] = crit.name
        out["parameter_price_nll"] = price
        out["n_merges_improving_score"] = sum(1 for r in ties if r["merge_improves_score"])
    return out


def fit_record(session):
    """The frozen record of the fit a diagnosis would describe, or ``None``.

    Every number in this module is a LOCAL expansion around a stationary point: the standard
    errors, the flat directions and, most sharply, the ONE-SHARED-VALUE merge price
    (:func:`merge_candidates`), which has no gradient term. Away from an optimum that price does
    not degrade gracefully, it gets monotonically cheaper — in one measured case, the fitted model
    prices its cheapest tie at 0.001 nll and the same structure at 3x its parameter values at
    7e-05 — so an un-fitted point advertises free merges that are not there.

    ``None`` therefore means "no optimum of this process's own to expand around": a session that
    has authored nothing, an ``auto_fit``-off commit, and a process that RESUMED a session. The last
    of those still HAS a fit to report on, so it is not left undiagnosed — the fit's own verdict
    travels with the fit (see :func:`fit_diagnosis_block`) and is re-served from there.
    """
    return getattr(session, "fit_record", None)


def structure_moved(session, structure_hash: str | None) -> bool:
    """Is the working structure a DIFFERENT one from ``structure_hash``?

    The one staleness question this module asks, of a frozen record and of a stored verdict alike —
    a COMPARISON, never a rebuild key. STRUCTURE (the model's equations plus the estimated-parameter
    set) is what decides whether a geometry describes the model in front of the agent. A restore
    points the working state at another version's model and does move it: the diagnosis then belongs
    to the last fit, says so, and is not re-priced (pricing the current structure would mean
    fitting, and building, it).

    The de-dup no-op deliberately does NOT move it. It re-authors the same structure with a new
    starting guess — which is why ``problem_hash`` differs and why no fit ran — so the frozen optimum
    is still HEAD's, still this structure's, and still the only fit there is to report on. Calling
    that stale is what silently switched off a client's parsimony finish gate on the one drift an
    agent can cause inside a session: the same rank-deficient FIM, refused before the no-op and
    ignored after it. A table-only drift is reported instead (see :data:`WORKING_TABLE_NOTE`).

    An unknown ``structure_hash`` reads as moved: unable to say the geometry still applies is not the
    same as saying it does.
    """
    if not structure_hash:
        return True
    try:
        from .parameters import structure_fingerprint

        return (
            structure_fingerprint(session.problem.model.sbml_model, session.problem.parameter_df)
            != structure_hash
        )
    except Exception:  # noqa: BLE001 - cannot tell: say so rather than claim currency
        return True


def is_stale(session, record) -> bool:
    """:func:`structure_moved` for a frozen record — has the fit it describes been superseded by a
    different structure in the working state?"""
    return structure_moved(session, getattr(record, "structure_hash", None))


def working_table_moved(session, record) -> bool:
    """Has the working PARAMETER TABLE moved since the fit, with the structure intact? (The de-dup
    no-op's new starting guess, or a bounds edit — worth saying, since the numbers reported are the
    fit's, not the table's.)"""
    try:
        return session.problem_hash() != record.problem_hash
    except Exception:  # noqa: BLE001 - cannot tell: prefer saying the values may have moved
        return True


#: Why a diagnosis was not computed. Both are reported in the ``available: False`` shape a missing
#: FIM already produces, so every consumer's existing fail-open path covers them.
NO_OPTIMUM_NOTE = (
    "no fitted optimum to diagnose — every number here is a local expansion around "
    "one, so none was computed. Commit a model with make_changes (which fits it) "
    "and read this again."
)
FAILED_NOTE = (
    "the parameter diagnosis could not be computed for this fit (see the server log); "
    "the residual section is unaffected."
)
#: Said when the working model's STRUCTURE has changed since the fit this describes. Reported, never
#: re-priced: pricing the current structure would mean fitting (and building) it.
STALE_NOTE = (
    "this describes the LAST FIT, not the model in front of you — a different structure is "
    "in the working state now (a restore). Commit the current model with make_changes "
    "(which fits it) to get its own diagnosis."
)
#: Said when the working parameter TABLE has moved but the structure has not — a restored version of
#: the same structure with other values. The geometry still describes this structure at its fitted
#: optimum, which is the only fit there is: nothing has been fitted on the table in front of you.
WORKING_TABLE_NOTE = (
    "your working parameter table is not the one this fit left (a value, a bound or "
    "a scale has changed since), and nothing has been fitted on it: these numbers "
    "are the fitted optimum of the SAME structure."
)
#: Said on a hierarchical fit, where the objective inner-solves the noise sigmas / observable
#: scalings and the FIM has no coordinate for them. Stated because their ABSENCE from `parameters`,
#: `n_estimated` and every priced tie would otherwise read as a bug rather than as the intent.
INNER_OMITTED_NOTE = (
    "this fit inner-solved its noise sigmas / observable-scaling factors, so they "
    "are not coordinates of this FIM: `n_estimated` here counts the OUTER "
    "parameters (fewer than the model's k) and no tie is priced against them. "
    "They are shipped by the observation model, so they define no merge you "
    "could write anyway."
)


def unavailable(note: str) -> dict[str, Any]:
    """The no-diagnosis payload, saying WHY (see :data:`NO_OPTIMUM_NOTE`, :data:`FAILED_NOTE`)."""
    return {"available": False, "method": "none", "note": note}


def _params_meta(record) -> list[dict[str, Any]]:
    """One ``{id, scale, estimate, estimate_scaled, at_bound}`` per free parameter, off the RECORD.

    The point, its scale and its box all come from the frozen problem, in the FIM's own coordinate
    order — not from ``parameter_df[nominalValue]``, which an agent edit can move after the fit, and
    which would then have a diagnosis expanding around a point the fit never reached.
    """
    from .problem import unscale

    out = []
    for i, name in enumerate(record.x_names):
        est_s = record.x_free[i]
        sc = record.scales[i]
        lb = record.lb[i] if i < len(record.lb) else None
        ub = record.ub[i] if i < len(record.ub) else None
        at_bound = False
        if lb is not None and ub is not None and ub > lb:
            # Same relative tolerance the fit report uses (fitting.parameters_at_bound), so a
            # parameter isn't flagged at-bound by one path and not the other.
            tol = PARAMETER_AT_BOUND_RELATIVE_TOL * (ub - lb)
            at_bound = (est_s - lb) <= tol or (ub - est_s) <= tol
        try:
            estimate = float(unscale(est_s, sc))
        except Exception:  # noqa: BLE001 - an unknown scale still has a scaled coordinate
            estimate = None
        out.append(
            {
                "id": name,
                "scale": sc,
                "estimate": estimate,
                "estimate_scaled": est_s,
                "at_bound": at_bound,
            }
        )
    return out


def identifiability(session, *, top_n: int = 8, corr_threshold: float = 0.9) -> dict[str, Any]:
    """Parameter-identifiability diagnosis of the last fit — which estimated parameters are
    unconstrained or trade off (the basis for model simplification). Combines the FIM (SE +
    marginal/partial correlations + flat directions) with the multistart spread. Bounded output.

    Reads ONLY the frozen :class:`fitting.FitRecord`, or — where this process ran no fit — the verdict
    the fit that DID run stored beside its score (:func:`stored_diagnosis`). No AMICI output folder is
    derived from live session state either way, so this cannot compile anything (see the module
    docstring). ``available: False`` when there is neither; ``stale: True`` when the working structure
    is no longer the one that was fitted.

    Which source, when both exist, is decided by WHICH FIT the working state is: the record while it
    describes that structure, and otherwise the stored verdict of the structure now in place — after a
    restore that is HEAD's own fit, and reporting the last fit's geometry there instead would make the
    section disagree with the ``merge_hints`` beside the score, which are read from the same store.
    """
    record = fit_record(session)
    if record is None or is_stale(session, record):
        stored = stored_diagnosis(session)
        if record is None or stored.get("available"):
            return stored
    # FIM first — its free-parameter order is the record's, which is the canonical order here.
    # Under the session lock (an RLock, so the commit path already holding it is unaffected):
    # building it flips the AMICI sigma-residual flag for one evaluation, and the objective it
    # flips it on is the engine's own cached one (`fitting.estimation_problem`), which every
    # simulate goes through. See ``docs/RATIONALE.md``.
    with session.lock:
        fim = _compute_fim(record)
    spread = _multistart_spread(record.optimization, list(record.x_names))
    # Roles keep the merge offers to edits a modeller would actually write: nothing mathematically
    # stops tying a kinetic rate to an observable scaling, and the arithmetic will call it free.
    # `fixed_tables` keeps them to edits a modeller CAN write: a parameter only the fixed tables
    # name (an observation offset, a noise sigma) is in no model text to rename.
    # merge_candidates is their only consumer and needs the FIM, so without one this whole-model
    # walk would be discarded unused. Walked on the FROZEN problem — the structure that was fitted.
    roles = fixed_tables = None
    if fim is not None:
        from .reconcile import fixed_table_parameters, parameter_roles

        try:
            roles = parameter_roles(record.problem)
        except Exception:  # noqa: BLE001 - advisory; the payload says the filter is off
            roles = None
        try:
            fixed_tables = fixed_table_parameters(record.problem)
        except Exception:  # noqa: BLE001 - advisory; the payload says the filter is off
            fixed_tables = None
    out = _identifiability_stats(
        _params_meta(record),
        fim,
        spread,
        top_n=top_n,
        corr_threshold=corr_threshold,
        roles=roles,
        fixed_tables=fixed_tables,
        n_data=_measurement_count(record.problem),
        criterion=session.criterion,
    )
    if record.hierarchical:
        out["inner_parameters_omitted"] = INNER_OMITTED_NOTE
    if is_stale(session, record):
        out["stale"] = True
        out["stale_note"] = STALE_NOTE
    elif working_table_moved(session, record):
        out["working_table_note"] = WORKING_TABLE_NOTE
    return out


#: How many ties ride beside a fit's score. Every one shown is a merge the criterion already pays
#: for, so the
#: cap is not about which ties qualify but about what the score line can carry: this list is re-sent
#: in the agent's history on every later turn (PR #23). Three is what it takes to stop the caller
#: merging PIECEMEAL — one pair per fit, a whole refit spent per deleted parameter, when the ties
#: were priced together and could be taken together.
MERGE_HINT_LIMIT = 3


def merge_hints(
    diagnosis: "dict[str, Any] | None", *, limit: int = MERGE_HINT_LIMIT, rename: bool = False
) -> list[dict[str, Any]] | None:
    """The ties worth reporting beside a fit's score: up to ``limit`` cheapest ONE-SHARED-VALUE
    merges whose ``dnll`` is below what the criterion charges for the parameter each deletes
    — or ``None``.

    ``diagnosis`` is an :func:`identifiability` payload, so the hints cannot say anything the
    section does not: they ARE that payload's ONE-SHARED-VALUE ``merge_candidates`` rows that beat
    the charge, cheapest first, minus the detail. Four fields each (and ``take``, below), kept
    short because this rides on every fit result and is re-sent in the agent's history every later
    turn: the pair, its cost, ``dscore`` (negative = the merge lowers the score), and
    ``move_scaled``. The last one is not decoration —
    :func:`merge_candidates` prices a ONE-SHARED-VALUE tie and says that price must be read against
    the move: a hint naming a near-free tie without it invites taking a merge whose parameter
    travels many decades, which is the quadratic running off down a flat direction rather than an
    edit that survives a refit.

    ALL of them, not just the cheapest, because one hint per fit taught the caller to merge one pair
    per refit while the rest of the redundancy sat unnamed — and on this problem class the ties come
    in GROUPS (a saturated model's parameters are jointly redundant), so naming one is
    naming an arbitrary member of a set the caller should be looking at whole.

    Each price is still an INDEPENDENT one, computed for that tie alone at this optimum: the list is
    not a menu whose ``dscore`` values add up. Rows sharing no parameter are separable edits and may be
    taken together; rows SHARING one are not. What is priced is the ONE SHARED VALUE of a PAIR —
    :func:`merge_candidates` prices ``theta_i == theta_j`` and no higher-order constraint — so tying
    three symbols to one value is an unpriced extrapolation whose cost is not the sum of its pairs.
    Measured: three pairs each within 0.02 nll of free, collapsed into one symbol, cost +186 bic and
    reintroduced lack-of-fit. Take the cheapest pair of such a group and refit — the next fit prices
    what is left, and the criterion adjudicates each step.

    ``None`` covers every case the list has nothing to claim in — no FIM, no priceable pair, or no
    tie that beats the charge — so its ABSENCE is not a certificate of parsimony. Only the section
    distinguishes those (``n_ties_skipped``, ``filters_off``).

    ``rename`` (a session whose ``make_changes`` offers ``model_edits``, ``session.text_route``)
    adds ``take``, the ``model_edits`` op that makes the tie: a ``rename`` of the second onto the
    first, which keeps its fitted value (and is the one the fixed tables read, if either is). Without it blasi agents retyped rate laws by hand and lost
    a constant doing so (PEtab-MCP#283). The statements route ties by ``rebind``, row by row.
    """
    ties = ((diagnosis or {}).get("merge_candidates") or {}).get("ties") or []
    out = []
    for row in ties:  # cheapest-first; each is ONE SHARED VALUE, priced on its own
        if not (isinstance(row, dict) and row.get("merge_improves_score")):
            continue
        pair = list(row.get("parameters") or ())
        hint = {
            "parameters": pair,
            "dnll": row.get("dnll"),
            "dscore": row.get("dscore"),
            "move_scaled": row.get("move_scaled"),
        }
        if rename and len(pair) == 2:
            hint["take"] = {"op": "rename", "old": pair[1], "new": pair[0]}
        out.append(hint)
    return out[:limit] or None


#: Key under which a fit stashes :func:`fit_diagnosis_block` on its simulation payload. Not part of
#: any agent-facing payload: the tool serves the diagnosis itself, and the fit result the hint.
STORED_DIAGNOSIS_KEY = "parameter_diagnosis"


def fit_diagnosis_block(session) -> dict[str, Any] | None:
    """The diagnosis of the fit ``session`` has just frozen, in the form the FIT payload carries it:
    ``{structure_hash, diagnosis}``. ``None`` when there is nothing to describe.

    This is how the parameter verdict outlives the process that computed it, exactly as the residual
    verdict does (:func:`data_access.residual_structure_rollup` rides on the same payload, and the
    payload is snapshotted per version). A session RESUMED in a new process restores HEAD's payload
    but no :class:`fitting.FitRecord` — the optimizer ran elsewhere — and re-deriving one there
    means naming an AMICI output folder from live state, which is a full C++ build of a structure
    that was already built once, under a different SBML serialisation. Storing the verdict removes
    the need: nothing is recomputed, so nothing can be compiled, and a client's parsimony refusal
    at finish keeps working across a resume instead of quietly going dark for the rest of the
    session.

    ``structure_hash`` travels with it because a stored verdict has to be able to say whether it still
    applies (see :func:`structure_moved`); it is the record's, i.e. the structure that was fitted.

    Best-effort in the strong sense: ANY failure returns ``None``. A parameter-geometry advisory must
    never fail a commit, so it is guarded here rather than at each call site — and it needs no compile
    guard, since :func:`identifiability` evaluates the FROZEN problem, which is already built. It runs
    the whole diagnosis (the FIM is the cost either way: measured at 0.085 s on a k=11, n=252
    steady-state model, 6% of a single-start fit there) so the stored verdict, the hint on the result
    and the section the agent reads are one computation at one optimum and cannot disagree.
    """
    try:
        record = fit_record(session)
        if record is None or is_stale(session, record):
            return None
        diagnosis = identifiability(session)
        if not diagnosis.get("available"):
            return None
        return {"structure_hash": record.structure_hash, "diagnosis": diagnosis}
    except Exception:  # noqa: BLE001 - advisory; a fit is never failed by a diagnostic
        return None


def fit_merge_hints(session) -> list[dict[str, Any]] | None:
    """:func:`merge_hints` for the fit ``session`` has just frozen, for the fit result to carry —
    the rows of :func:`fit_diagnosis_block`'s verdict the result has room for."""
    return merge_hints(
        (fit_diagnosis_block(session) or {}).get("diagnosis"), rename=session.text_route
    )


def stored_diagnosis(session) -> dict[str, Any] | None:
    """The parameter diagnosis the FIT stored (:func:`fit_diagnosis_block`), re-served for a process
    that did not run that fit — or the ``available: False`` payload when there is none.

    Nothing is recomputed here: the stale label is re-evaluated against the working structure (the
    stored one was computed when that structure WAS the working one), and the rest is the fit's own
    verdict, computed at its own optimum. ``top_n`` therefore has no effect on it — a stored verdict
    is bounded the way the fit bounded it.
    """
    stored = (getattr(session, "simulation", None) or {}).get(STORED_DIAGNOSIS_KEY)
    out = _stored_verdict(stored)
    if out is None:
        return unavailable(NO_OPTIMUM_NOTE)
    if structure_moved(session, stored.get("structure_hash")):
        out["stale"] = True
        out["stale_note"] = STALE_NOTE
    return out


#: Why ``get_fit_diagnosis(version=N)`` has no parameter section for that version.
NO_STORED_DIAGNOSIS_NOTE = (
    "version {version} stored no parameter diagnosis: its fit froze no optimum, or the FIM "
    "there could not be computed. A read never refits to make one."
)


def version_diagnosis(simulation: dict[str, Any] | None, version: int) -> dict[str, Any]:
    """Stored ``version``'s parameter diagnosis, off the ``simulation`` blob it stored
    (:func:`fit_diagnosis_block`), or the ``available: False`` payload saying there is none.

    Nothing is recomputed, so ``top_n`` has no effect, and nothing is labelled stale: it describes
    that version's own fit, whatever the working state is now."""
    out = _stored_verdict((simulation or {}).get(STORED_DIAGNOSIS_KEY))
    if out is None:
        return unavailable(NO_STORED_DIAGNOSIS_NOTE.format(version=version))
    return out


def _stored_verdict(stored: Any) -> dict[str, Any] | None:
    """The verdict of a stored :func:`fit_diagnosis_block` without the labels that compare it to
    the working state then, or ``None`` where it holds no available one."""
    diagnosis = stored.get("diagnosis") if isinstance(stored, dict) else None
    if not isinstance(diagnosis, dict) or not diagnosis.get("available"):
        return None
    return {
        k: v for k, v in diagnosis.items() if k not in ("stale", "stale_note", "working_table_note")
    }


def stored_merge_hints(session) -> list[dict[str, Any]] | None:
    """The hints the last fit already priced, for a later READ of the same score
    (:func:`fitting.fit_summary`) to report without pricing them a second time — or ``None``.

    Read off the STORED verdict, never recomputed: the score read is the cheapest call in the
    session and a client's finish gate makes it on every finish attempt, so it must not cost a FIM.
    ``None`` once the STRUCTURE has drifted from the fit that priced them, so a restore does not
    re-serve another model's ties as if they were this state's — but a de-dup no-op does not drop
    them: HEAD is still that fit, and the score read beside it is still that fit's score.
    """
    diagnosis = stored_diagnosis(session)
    if not diagnosis.get("available") or diagnosis.get("stale"):
        return None
    return merge_hints(diagnosis, rename=session.text_route)
