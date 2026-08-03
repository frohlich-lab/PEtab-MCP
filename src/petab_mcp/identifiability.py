"""Parameter-identifiability diagnosis of the current fit.

Which estimated parameters are superfluous? The lead over-parameterises because raw nll/bic
don't say WHICH parameters are unconstrained or trading off. ``get_identifiability`` (see
:mod:`petab_mcp.server`) answers that from two cheap, complementary signals at the fitted
optimum:

* the Fisher Information Matrix (FIM ≈ Hessian of the nllh, from AMICI sensitivities) ->
  per-parameter standard error, marginal + partial correlations, and — via its
  eigendecomposition — the non-identifiable parameter COMBINATIONS (flat directions);
* the multistart spread: how much each parameter wanders across near-optimal starts (a
  model-free practical-identifiability check that needs no Hessian).

The FIM is a LOCAL linearization and can understate non-identifiability for strongly nonlinear
models, so the two are meant to be read together. All output is bounded (top-N pairs/directions).

Split out of :mod:`petab_mcp.fitting` (which owns fit/simulate orchestration) so this
analytical subsystem stands on its own; the multistart spread reads
``session.optimization`` written by the fitter.
"""

from __future__ import annotations

import math
from typing import Any

from petab.v1.C import ESTIMATE, NOMINAL_VALUE, PARAMETER_SCALE

from .constants import PARAMETER_AT_BOUND_RELATIVE_TOL
from .parameters import estimated_mask


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
    n_data: int | None = None,
) -> dict[str, Any]:
    """Pure statistics core (no pyPESTO/AMICI). ``params_meta`` is one dict per estimated
    parameter — ``{id, scale, estimate, estimate_scaled, at_bound}`` — in the SAME order as the
    rows/cols of ``fim`` (an ``m×m`` numpy array or ``None``). ``spread`` maps parameter id ->
    ``{std,min,max,n}`` from the multistart. Returns the compact, bounded diagnosis dict."""
    import numpy as np

    names = [p["id"] for p in params_meta]
    m = len(names)
    per = {p["id"]: {
        "id": p["id"], "scale": p.get("scale"), "estimate": p.get("estimate"),
        "at_bound": bool(p.get("at_bound")), "se_scaled": None, "fold_uncertainty": None,
        "multistart_std": None, "multistart_range": None,
    } for p in params_meta}

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
                    correlations.append({"a": names[i], "b": names[j], "r": round(float(marg[i, j]), 3)})
                if abs(pcorr[i, j]) >= corr_threshold:
                    partials.append({"a": names[i], "b": names[j], "pr": round(float(pcorr[i, j]), 3)})
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
                participants = [{"id": names[k], "weight": round(float(vec[k]), 3)}
                                for k in order[:4] if abs(vec[k]) > 0.15]
                flat_directions.append({"eigenvalue": round(ev, 6), "rel": round(rel, 6),
                                        "participants": participants})
        rank = int(np.linalg.matrix_rank(H))
        cond = float(emax / max(float(evals[evals > 0].min()) if (evals > 0).any() else emax, 1e-300))
        fim_info = {"rank": rank, "n_estimated": m, "condition": round(cond, 1),
                    "rank_deficient": rank < m}

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
            reasons.append(f"multistart_unstable(std={round(p['multistart_std'],3)})")
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
                fim, names, theta,
                scales=[p.get("scale") for p in params_meta],
                at_bound=[per[n]["at_bound"] for n in names],
                roles=roles, n_data=n_data, top_n=top_n)

    method = ("fim+multistart" if fim is not None and spread else
              "fim" if fim is not None else
              "multistart" if spread else "none")
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
        "notes": ("FIM is a LOCAL linearization of uncertainty — read se/correlations together "
                  "with multistart_std (a model-free spread) and flat_directions (the actual "
                  "non-identifiable parameter combinations). High |r|≈1, large fold_uncertainty, "
                  "at_bound, or a flat direction all mark a parameter as a simplification "
                  "candidate: remove its term, take its limiting (irreversible/instantaneous) "
                  "form, or merge a collinear pair into one parameter — all MODEL edits. "
                  "`merge_candidates` prices the ONE-SHARED-VALUE form of that merge, cheapest "
                  "first, with the parameter move each tie would take. "
                  "If none fits cleanly, leave it: there is no value/estimate "
                  "tool, and a hard-coded literal is a magic number; values come only from the fit."),
    }


def _compute_fim(session, engine):
    """FIM (≈ Hessian of the nllh) at the current fitted point, in free-scaled parameter space.

    Returns ``(fim_ndarray, free_names)`` or ``None`` (best-effort — a missing FIM just drops
    the correlation half of the diagnosis, leaving the multistart spread)."""
    try:
        import numpy as np
        import pypesto.petab
        from petab.v1.parameters import scale

        from .engine import _ensure_swig_on_path
        _ensure_swig_on_path()
        importer = pypesto.petab.PetabImporter(
            session.problem, **engine.pypesto_importer_kwargs()
        )
        problem = importer.create_problem()
        df = session.problem.parameter_df
        x_full = []
        for name in problem.x_names:
            if name in df.index:
                x_full.append(float(scale(float(df.loc[name, NOMINAL_VALUE]),
                                          df.loc[name, PARAMETER_SCALE])))
            else:
                x_full.append(0.0)
        free_idx = list(problem.x_free_indices)
        free_names = [problem.x_names[i] for i in free_idx]
        x_free = np.asarray(problem.get_reduced_vector(np.asarray(x_full, dtype=float)), dtype=float)
        mfree = len(free_names)

        def _finite_square(a):
            a = np.asarray(a, dtype=float)
            return a.shape == (mfree, mfree) and np.isfinite(a).all()

        # Preferred: the objective's Hessian (AMICI returns the FIM approximation for a
        # Gaussian LSQ). Fallback: build the FIM from residual sensitivities sres^T sres.
        try:
            ret = problem.objective(x_free, sensi_orders=(0, 1, 2), return_dict=True)
            for key in ("hess", "HESS"):
                if key in ret and _finite_square(ret[key]):
                    return np.asarray(ret[key], dtype=float), free_names
        except Exception:  # noqa: BLE001 - order-2 unsupported for this objective; try sres
            pass
        try:
            ret = problem.objective(x_free, sensi_orders=(0, 1), return_dict=True)
            for key in ("sres", "SRES"):
                if key in ret and ret[key] is not None:
                    sres = np.asarray(ret[key], dtype=float)
                    fim = sres.T @ sres
                    if _finite_square(fim):
                        return fim, free_names
        except Exception:  # noqa: BLE001
            pass
        return None
    except Exception:  # noqa: BLE001 - importer/model build failure; degrade to spread-only
        return None


def _multistart_spread(session, free_names: list[str]) -> dict[str, Any]:
    """Per-parameter spread across near-optimal multistarts (scaled space), from the stored
    optimization payload's per-start ``x``. Empty when x/x_names were not recorded."""
    import numpy as np

    opt = getattr(session, "optimization", None) or {}
    starts = opt.get("starts") or []
    xnames = opt.get("x_names") or []
    if not starts or not xnames:
        return {}
    col = {n: i for i, n in enumerate(xnames)}
    fvals = [float(s["fval"]) for s in starts
             if s.get("fval") is not None and math.isfinite(float(s["fval"]))]
    if not fvals:
        return {}
    best = min(fvals)
    # near-optimal band; fall back to the best few starts if the band is too tight
    band = [s for s in starts if s.get("fval") is not None and s.get("x") is not None
            and math.isfinite(float(s["fval"])) and float(s["fval"]) <= best + 1.0]
    if len(band) < 3:
        band = sorted([s for s in starts if s.get("x") is not None and s.get("fval") is not None],
                      key=lambda s: float(s["fval"]))[:5]
    out: dict[str, Any] = {}
    for n in free_names:
        j = col.get(n)
        if j is None:
            continue
        vals = [float(s["x"][j]) for s in band if s.get("x") is not None and j < len(s["x"])]
        if len(vals) >= 2:
            arr = np.asarray(vals)
            out[n] = {"std": float(arr.std()), "min": float(arr.min()),
                      "max": float(arr.max()), "n": len(vals)}
    return out


def _measurement_count(session) -> int | None:
    """Number of measurement rows — the ``n`` in BIC's ``k*ln(n)`` penalty, so a tie's cost can be
    compared against what BIC charges for the parameter it would remove."""
    try:
        n = len(session.problem.measurement_df)
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


def merge_candidates(fim, names, theta, *, scales=None, at_bound=None, roles=None,
                     n_data=None, top_n=6):
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
    price it honestly — same ROLE (nothing stops the arithmetic tying a rate to a noise sigma),
    same parameterScale (``log10(k_a) == k_b`` is not an equality anyone means), and neither member
    AT A BOUND (the expansion assumes an interior stationary point, so a pinned parameter's cost
    would be understated). Each filter needs its table: whichever the caller leaves out is named in
    ``filters_off``, and every pair a filter refused is counted by reason in ``n_ties_skipped``, so
    that "no free tie exists" and "the free tie was suppressed" read differently. The ranked list is
    truncated to ``top_n``.

    Read as a SCREEN. The quadratic is local: it separates cheap ties from expensive ones, but a
    single price can be off by a factor in EITHER direction, so BIC adjudicates the refit that
    follows a merge.
    """
    import numpy as np

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
    curv = d[:, None] + d[None, :] - 2.0 * Hinv   # c @ inv(H) @ c for every c = e_i - e_j at once
    gap = th[:, None] - th[None, :]               # c @ theta likewise

    price = round(math.log(n_data) / 2.0, 3) if n_data and n_data > 0 else None
    ties: list[dict[str, Any]] = []
    skipped: dict[str, int] = {}

    def _skip(reason):
        skipped[reason] = skipped.get(reason, 0) + 1

    for i in range(k):
        for j in range(i + 1, k):
            # Each filter FAILS CLOSED, and says which way it closed: "these two differ" is a claim
            # about the pair, "this one is unknown" is a claim about the table, and a reader acts on
            # them differently.
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
                # largest component, the move the panel would have to believe.
                move = float(abs(gap[i, j]) * np.abs(Hinv[:, i] - Hinv[:, j]).max() / curv[i, j])
            # 2*dnll is the BIC charge, so a cost within a factor of two of the double ceiling is
            # `Infinity` in the payload — not valid strict JSON, and not a merge candidate either.
            if not (math.isfinite(2.0 * exact) and math.isfinite(move)):
                _skip("cost_not_finite")
                continue
            dnll = _cost_nll(exact)
            row = {"parameters": [names[i], names[j]], "dnll": dnll,
                   "move_scaled": round(move, 3)}
            if price is not None:
                # Both read off the DISPLAYED cost and price, so the row agrees with itself:
                # dbic < 0 exactly when merge_improves_bic is true.
                row["dbic"] = round(2.0 * (dnll - price), 3)
                row["merge_improves_bic"] = dnll < price
            ties.append(row)
    # Cheapest first, and among the ties the price cannot separate — the ones the data left
    # undetermined, all of them free — the smallest move first, since that is what decides which of
    # them is a writable edit. Both keys are on the row, so the order can be checked by reading it.
    ties.sort(key=lambda r: (r["dnll"], r["move_scaled"]))
    out: dict[str, Any] = {"ties": ties[:top_n], "n_ties_priced": len(ties)}
    if skipped:
        out["n_ties_skipped"] = dict(sorted(skipped.items()))
    off = [n for n, table in (("role", roles), ("scale", scales), ("at_bound", at_bound))
           if table is None]
    if off:  # nothing refused these pairs, so merges nobody could write are in the list too
        out["filters_off"] = off
    if price is not None:
        out["bic_price_nll"] = price
        out["n_merges_improving_bic"] = sum(1 for r in ties if r["merge_improves_bic"])
    return out


def identifiability(session, engine=None, *, top_n: int = 8, corr_threshold: float = 0.9) -> dict[str, Any]:
    """Parameter-identifiability diagnosis of the current fit — which estimated parameters are
    unconstrained or trade off (the basis for model simplification). Combines the FIM (SE +
    marginal/partial correlations + flat directions) with the multistart spread. Bounded output.
    """
    from petab.v1.C import LOWER_BOUND, UPPER_BOUND
    from petab.v1.parameters import scale

    df = session.problem.parameter_df
    engine = engine or getattr(session, "engine", None)

    # FIM first — its free-parameter order is the canonical order for everything else.
    fim = free_names = None
    if engine is not None:
        got = _compute_fim(session, engine)
        if got is not None:
            fim, free_names = got
    if free_names is None:  # no FIM: use the estimated params in table order
        free_names = []
        if ESTIMATE in df.columns:
            # `estimate` can be NaN / non-numeric; estimated_mask coerces defensively (a bad cell
            # yields "not estimated") and here a wholly non-numeric column falls back to none free.
            try:
                free_names = [str(i) for i in df.index[estimated_mask(df)]]
            except (TypeError, ValueError):
                free_names = []

    def _scaled(name, col):
        try:
            return float(scale(float(df.loc[name, col]), df.loc[name, PARAMETER_SCALE]))
        except Exception:  # noqa: BLE001 - missing bound/value
            return None

    params_meta = []
    for name in free_names:
        est_s = _scaled(name, NOMINAL_VALUE) if name in df.index else None
        lb = _scaled(name, LOWER_BOUND) if (name in df.index and LOWER_BOUND in df.columns) else None
        ub = _scaled(name, UPPER_BOUND) if (name in df.index and UPPER_BOUND in df.columns) else None
        at_bound = False
        if est_s is not None and lb is not None and ub is not None and ub > lb:
            # Same relative tolerance the fit report uses (fitting.parameters_at_bound), so a
            # parameter isn't flagged at-bound by one path and not the other.
            tol = PARAMETER_AT_BOUND_RELATIVE_TOL * (ub - lb)
            at_bound = (est_s - lb) <= tol or (ub - est_s) <= tol
        params_meta.append({
            "id": name,
            "scale": (df.loc[name, PARAMETER_SCALE] if name in df.index else None),
            "estimate": (float(df.loc[name, NOMINAL_VALUE]) if name in df.index else None),
            "estimate_scaled": est_s,
            "at_bound": at_bound,
        })

    spread = _multistart_spread(session, free_names)
    # Roles keep the merge offers to edits a modeller would actually write: nothing mathematically
    # stops tying a kinetic rate to an observable scaling, and the arithmetic will call it free.
    # merge_candidates is their only consumer and needs the FIM, so without one this whole-model
    # walk would be discarded unused.
    roles = None
    if fim is not None:
        try:
            from .reconcile import parameter_roles
            roles = parameter_roles(session.problem)
        except Exception:  # noqa: BLE001 - advisory; the payload says the filter is off
            roles = None
    return _identifiability_stats(params_meta, fim, spread,
                                  top_n=top_n, corr_threshold=corr_threshold,
                                  roles=roles, n_data=_measurement_count(session))
