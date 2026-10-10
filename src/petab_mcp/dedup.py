"""Structure-level de-duplication for the ``fit`` tool (issue #59; relocated by issue #81).

The model-building loop's single most frequent waste is fitting a structure it has already
visited: a byte-identical re-encode of the current model (same reactions, same fit, zero change),
or oscillating back into a structure family that already scored worse — each one recompiles and
refits to the *same* result. This module gives ``fit`` a memory. It fingerprints the resolved
structure (see :func:`petab_mcp.parameters.structure_fingerprint`) and, BEFORE the expensive
recompile+fit, decides whether to

* **skip** a no-op fit (byte-identical to the current version) — no fit, no new version;
* **warn** when revisiting a structure already in the history, or a family whose best score was
  materially worse than the best model so far (the best model is kept and scored regardless);
* **terminate** an identical-fit loop after a few repeats (they cannot change anything).

The heavy lifting is the pure :func:`decide` — it takes fingerprints + a small registry of
prior versions and returns an action, so it is trivially testable without AMICI. The caller
(:meth:`petab_mcp.session.PetabSession.fit`) computes the fingerprints, runs :func:`decide`, and
enacts the result. See issues #59 and #81.
"""

from __future__ import annotations

from typing import Any

from .formatting import finite_float
from .parameters import estimated_ids

# The same structure may be authored at most this many times before ``set_model`` refuses
# further identical re-authors: an identical structure recompiles + refits to the SAME result
# and cannot improve the fit, so a run that keeps emitting it is stuck. The way out is a
# genuinely different structural change; the best model is kept and scored regardless, so there
# is nothing to go back to.
MAX_IDENTICAL_REAUTHORS = 3

# Warn when re-authoring a structure family whose best (lowest) score was at least this much
# WORSE (higher) than the best score seen so far: revisiting it is very unlikely to beat the
# current best. ~5 units is the conventional "positive evidence" threshold; every criterion is on
# the same deviance scale (see :mod:`criteria`), so it holds for whichever one is active.
WORSE_SCORE_MARGIN = 5.0


def estimation_setup_signature(parameter_df) -> tuple:
    """A hashable signature of the parameter table's ESTIMATION setup — everything that
    determines the fit EXCEPT an estimated parameter's starting value.

    For each parameter: ``(id, estimated, scale, lowerBound, upperBound)``, plus the
    ``nominalValue`` for FIXED parameters only (a fixed value enters the fit; an estimated
    one's is just a starting guess the multistart ignores and the fit overwrites). Two tables
    with the same signature yield the same fit, so comparing a re-authored candidate's
    signature to the current version's is the safe test for "this changes nothing".
    """
    if parameter_df is None:
        return ()
    from petab.v2.C import LOWER_BOUND, NOMINAL_VALUE, UPPER_BOUND

    from .problem import PARAMETER_SCALE

    df = parameter_df
    cols = getattr(df, "columns", [])
    estimated = set(estimated_ids(df))
    rows: list[tuple] = []
    for pid in sorted(str(p) for p in df.index):
        row = df.loc[pid]
        est = pid in estimated
        scale = str(row[PARAMETER_SCALE]) if PARAMETER_SCALE in cols else None
        lb = finite_float(row[LOWER_BOUND]) if LOWER_BOUND in cols else None
        ub = finite_float(row[UPPER_BOUND]) if UPPER_BOUND in cols else None
        entry: tuple = (pid, est, scale, lb, ub)
        if not est:
            entry = entry + (finite_float(row[NOMINAL_VALUE]) if NOMINAL_VALUE in cols else None,)
        rows.append(entry)
    return tuple(rows)


def reauthor_key(structure_hash: str, parameter_df) -> str:
    """Identity of an *identical re-author* — the same structure AND the same estimation setup,
    so the fit is fully determined and would reproduce byte-for-byte.

    The re-author loop guard counts attempts under this key rather than ``structure_hash`` alone.
    That distinction matters: ``structure_hash`` deliberately ignores bounds/scale, so counting on
    it would also count re-authoring the SAME structure with DIFFERENT bounds/scale — a legitimate
    change that CAN move the fit (e.g. iterative bound widening) — and could refuse it. Keying on
    (structure + setup) means only a genuinely identical re-author (which truly cannot change the
    result) is counted toward termination.
    """
    from .parameters import sha16

    return sha16(structure_hash + "\n" + repr(estimation_setup_signature(parameter_df)))


def build_registry(session) -> list[dict[str, Any]]:
    """Prior FITTED versions as ``{version, structure_hash, neg_log_likelihood, score, criterion}``
    dicts, read from each version's stored meta.

    ``score`` is the version's value under the run's criterion (see
    :meth:`criteria.Criterion.score_of`) — ``None`` for a version stored without one, which just
    leaves it out of the ranking — and ``criterion`` is that criterion's label, carried ON the
    entry so :func:`decide` can name it without a session in hand.

    Every version is a fit, so every version competes. Versions written before the
    ``structure_hash`` field existed are skipped — they simply do not participate in
    de-duplication. Never raises: bookkeeping must not break a commit.
    """
    registry: list[dict[str, Any]] = []
    criterion = session.criterion
    try:
        metas = session.provenance.list_versions()
    except Exception:  # noqa: BLE001 - registry is advisory; a read failure just disables dedup
        return registry
    for meta in metas:
        sh = meta.get("structure_hash")
        if not sh:
            continue
        registry.append(
            {
                "version": meta.get("version"),
                "structure_hash": sh,
                "neg_log_likelihood": meta.get("neg_log_likelihood"),
                "score": criterion.score_of(meta),
                "criterion": criterion.name,
            }
        )
    return registry


def _min_score_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The best-scoring (lowest) entry, or ``None``. Uses :func:`~petab_mcp.formatting.finite_float`
    (not an ``isinstance`` check) so a score stored as a NumPy scalar — common in fit payloads —
    still counts; an ``isinstance(x, float)`` filter would silently drop ``numpy.float64`` and
    quietly disable the worse-family / best-structure logic."""
    scored = [e for e in entries if finite_float(e.get("score")) is not None]
    return min(scored, key=lambda e: finite_float(e["score"])) if scored else None


def decide(
    *,
    structure_hash: str,
    head_structure_hash: str | None,
    same_setup: bool,
    head_version: int,
    registry: list[dict[str, Any]],
    prior_attempts: int,
) -> dict[str, Any]:
    """Decide what a ``set_model`` re-author should do — run AFTER reconcile+validate, BEFORE
    the fit. Pure: all state comes in as arguments.

    Arguments:
        structure_hash: the candidate's :func:`structure_fingerprint`.
        head_structure_hash: the current HEAD version's structure fingerprint (``None`` if none).
        same_setup: whether the candidate's estimation setup (bounds/scale/estimate + fixed
            values) is identical to HEAD's — see :func:`estimation_setup_signature`.
        head_version: the current HEAD version number.
        registry: prior committed versions (see :func:`build_registry`).
        prior_attempts: how many times this exact structure has already been authored this run.

    Returns one of:
        ``{"action": "terminate", "message": ...}`` — refuse; the run is looping on one structure.
        ``{"action": "skip", "duplicate_of": v, "message": ..., "warnings": [...]}`` —
            byte-identical to the current HEAD: skip the recompile+fit and the new version.
        ``{"action": "proceed", "warnings": [...]}`` — commit normally, attaching any advisories.
    """
    # (d) An identical structure fit too many times cannot change anything — stop the loop.
    if prior_attempts >= MAX_IDENTICAL_REAUTHORS:
        # `prior_attempts` fits actually RAN; this call is the one being refused, so it is not
        # among them (saying "fit N+1 times now" claimed a fit that never happened).
        return {
            "action": "terminate",
            "message": (
                f"This same model structure has already been fit {prior_attempts} times. Re-fitting an "
                "identical structure recompiles and refits to the SAME result — it cannot improve the "
                "fit. Stop re-fitting it: your best model is kept and scored regardless (see "
                "list_versions), so make a genuinely different structural change and fit THAT."
            ),
        }

    # (a) Byte-identical to the current HEAD (same structure AND same estimation setup): nothing
    # would change — skip the recompile+fit+snapshot and report the no-op.
    if head_structure_hash is not None and structure_hash == head_structure_hash and same_setup:
        return {
            "action": "skip",
            "duplicate_of": head_version,
            "warnings": [],
            "message": (
                f"This model is identical to the current version (v{head_version}); the fit was "
                "not re-run (its metrics are unchanged — see get_fit_summary). To improve the fit, "
                "change the STRUCTURE and fit again; an earlier model you already committed is "
                "kept and scored regardless."
            ),
        }

    warnings: list[str] = []
    matches = [r for r in registry if r.get("structure_hash") == structure_hash]
    # Prior versions with this structure, excluding the current HEAD (re-authoring HEAD's own
    # structure with new bounds is a deliberate edit, not a revisit to restore).
    prior_versions = sorted(
        v for r in matches if (v := r.get("version")) is not None and v != head_version
    )
    if prior_versions:
        warnings.append(
            f"This model structure was already committed at version(s) {prior_versions} and is "
            "kept and scored regardless; re-fitting it just creates a duplicate — make a genuinely "
            "different change instead."
        )
    # (b) The structure family already scored materially worse than the best model so far.
    family = _min_score_entry(matches)
    best = _min_score_entry(registry)
    if family is not None and best is not None:
        # Both are non-None: _min_score_entry only ever returns an entry whose score already
        # passed finite_float, so no second None-check is needed here.
        fb, gb = finite_float(family["score"]), finite_float(best["score"])
        if fb - gb > WORSE_SCORE_MARGIN:
            # Off the ENTRY, so the label cannot describe a different criterion than the two
            # numbers beside it. A registry built without one degrades to the generic word.
            name = best.get("criterion") or "score"
            warnings.append(
                f"This structure family previously scored {name}≈{fb:.1f}, {fb - gb:.1f} WORSE "
                f"than the best model so far ({name}≈{gb:.1f}, v{best.get('version')}). "
                f"Revisiting it is "
                f"unlikely to beat the best (v{best.get('version')} is kept and scored regardless) "
                "— try a different structure."
            )
    return {"action": "proceed", "warnings": warnings}
