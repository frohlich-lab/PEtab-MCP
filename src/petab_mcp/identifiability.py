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
                    per[pid]["fold_uncertainty"] = round(float(base ** se[i]), 3)
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
        "notes": ("FIM is a LOCAL linearization of uncertainty — read se/correlations together "
                  "with multistart_std (a model-free spread) and flat_directions (the actual "
                  "non-identifiable parameter combinations). High |r|≈1, large fold_uncertainty, "
                  "at_bound, or a flat direction all mark a parameter as a simplification "
                  "candidate: remove its term, take its limiting (irreversible/instantaneous) "
                  "form, or merge a collinear pair into one parameter (e.g. a ratio/product) — "
                  "all MODEL edits. If none fits cleanly, leave it: there is no value/estimate "
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
        h = session.sbml_hash()
        importer = pypesto.petab.PetabImporter(
            session.problem,
            output_folder=str(engine.build_root / f"pypesto_{h}"),
            model_name=f"petab_mcp_pp_{h}",
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
            # `estimate` can be NaN / non-numeric; coerce defensively (fillna(0)) so a bad cell
            # yields "not estimated" rather than crashing this FIM-less fallback with a
            # ValueError from int(NaN) — matching the fillna guard used elsewhere.
            try:
                est = df[ESTIMATE].fillna(0).astype(float)
                free_names = [str(i) for i in df.index if est.loc[i] != 0]
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
            tol = 1e-3 * (ub - lb)
            at_bound = (est_s - lb) <= tol or (ub - est_s) <= tol
        params_meta.append({
            "id": name,
            "scale": (df.loc[name, PARAMETER_SCALE] if name in df.index else None),
            "estimate": (float(df.loc[name, NOMINAL_VALUE]) if name in df.index else None),
            "estimate_scaled": est_s,
            "at_bound": at_bound,
        })

    spread = _multistart_spread(session, free_names)
    return _identifiability_stats(params_meta, fim, spread,
                                  top_n=top_n, corr_threshold=corr_threshold)
