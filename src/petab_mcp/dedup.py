"""Structure-level de-duplication for the ``fit`` tool (issue #59; relocated by issue #81).

The model-building loop's single most frequent waste is fitting a structure it has already
visited: a byte-identical re-encode of the current model (same reactions, same fit, zero change),
or oscillating back into a structure family that already scored worse — each one recompiles and
refits to the *same* result. This module gives ``fit`` a memory. It fingerprints the resolved
structure (see :func:`petab_mcp.parameters.structure_fingerprint`) and, BEFORE the expensive
recompile+fit, decides whether to

* **skip** a no-op fit (byte-identical to the current version) — no fit, no new version;
* **warn** when revisiting a structure already in the history, or a family whose best BIC was
  materially worse than the best model so far (and point at ``restore_version(n)``);
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
# and cannot improve the fit, so a run that keeps emitting it is stuck. The way out is
# ``restore_version(n)`` (if it is the best model, it is already committed) or a genuinely
# different structural change — both of which reset this loop.
MAX_IDENTICAL_REAUTHORS = 3

# Warn when re-authoring a structure family whose best (lowest) BIC was at least this much
# WORSE (higher) than the best BIC seen so far: revisiting it is very unlikely to beat the
# current best. ~5 BIC units is the conventional "positive evidence" threshold.
WORSE_BIC_MARGIN = 5.0


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
    from petab.v1.C import LOWER_BOUND, NOMINAL_VALUE, PARAMETER_SCALE, UPPER_BOUND

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
    """Prior committed versions as ``{version, structure_hash, problem_hash, neg_log_likelihood,
    bic}`` dicts, read from each version's stored meta.

    Versions written before the ``structure_hash`` field existed are skipped — they simply do
    not participate in de-duplication. Never raises: bookkeeping must not break a commit.
    """
    registry: list[dict[str, Any]] = []
    try:
        metas = session.provenance.list_versions()
    except Exception:  # noqa: BLE001 - registry is advisory; a read failure just disables dedup
        return registry
    for meta in metas:
        sh = meta.get("structure_hash")
        if not sh:
            continue
        registry.append({
            "version": meta.get("version"),
            "structure_hash": sh,
            "problem_hash": meta.get("problem_hash"),
            "neg_log_likelihood": meta.get("neg_log_likelihood"),
            "bic": meta.get("bic"),
        })
    return registry


def _min_bic_entry(entries: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The entry with the smallest finite BIC, or ``None``. Uses :func:`~petab_mcp.formatting.finite_float` (not an
    ``isinstance`` check) so a BIC stored as a NumPy scalar — common in fit payloads — still
    counts; an ``isinstance(x, float)`` filter would silently drop ``numpy.float64`` and quietly
    disable the worse-family / best-structure logic."""
    scored = [e for e in entries if finite_float(e.get("bic")) is not None]
    return min(scored, key=lambda e: finite_float(e["bic"])) if scored else None


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
        return {"action": "terminate", "message": (
            f"This same model structure has been fit {prior_attempts + 1} times now. Re-fitting an "
            "identical structure recompiles and refits to the SAME result — it cannot improve the "
            "fit. Stop re-fitting it: if this is your best model it is already committed, so go "
            "back to it with restore_version(n) (use list_versions to find the best-bic version), "
            "or make a genuinely different structural change and fit THAT."
        )}

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
                "change the STRUCTURE and fit again; to go back to an earlier model, use "
                "restore_version(n)."
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
        latest = prior_versions[-1]
        warnings.append(
            f"This model structure was already committed at version(s) {prior_versions}. "
            f"restore_version({latest}) brings that model back without recompiling or refitting; "
            "re-fitting it just creates a duplicate."
        )
    # (b) The structure family already scored materially worse than the best model so far.
    family = _min_bic_entry(matches)
    best = _min_bic_entry(registry)
    if family is not None and best is not None:
        fb, gb = finite_float(family["bic"]), finite_float(best["bic"])
        if fb is not None and gb is not None and fb - gb > WORSE_BIC_MARGIN:
            warnings.append(
                f"This structure family previously scored bic≈{fb:.1f}, {fb - gb:.1f} WORSE than "
                f"the best model so far (bic≈{gb:.1f}, v{best.get('version')}). Revisiting it is "
                f"unlikely to beat the best — consider restore_version({best.get('version')}) or a "
                "different structure."
            )
    return {"action": "proceed", "warnings": warnings}
