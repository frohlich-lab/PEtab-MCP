"""The constrained, justified model/parameter edits.

Structural change is made by authoring the whole model with ``set_model`` (the single model
editor); the parameter-table edits adjust a parameter's search space.

Every edit:

1. requires a non-empty ``justification`` (``references`` are optional; any given are
   validated via the session's :class:`ReferenceResolver`);
2. applies a structural edit to a *copy* of the model (libsbml);
3. reconciles SBML↔PEtab (parametric ICs, parameter-table sync, orphan pruning);
4. lints the candidate — on any failure the edit is reverted (atomic);
5. logs a provenance ``edit`` event and records the change as *pending a fit*.

Issue #81 decoupled editing from fitting: an edit mutates the working model but does NOT fit
and does NOT create a version. The explicit ``fit`` tool (see :meth:`session.PetabSession.fit`)
scores the assembled working model — it fits, simulates, and snapshots the single version that
records the fitted values, fit metrics, and simulation dataframe.
"""

from __future__ import annotations

import re
from typing import Any, Callable

from petab.v1.C import LOWER_BOUND, UPPER_BOUND
from petab.v1.models.sbml_model import SbmlModel

from . import reconcile as R
from .constants import NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD, WIDENING_NEGLIGIBLE_BIC_GAIN
from .errors import ModelUpdateError, PetabMcpError, ValidationError
from .parameters import estimated_mask, problem_fingerprint, sha16
from .references import resolve_references
from .validation import (
    frozen_parametric_stoichiometry_issues,
    stoichiometry_conflict_issues,
    validate_problem,
)

# AMICI reserves a handful of short symbols for its generated code (the state vector ``x``,
# parameters ``p``, constants ``k``, expressions ``w``, time ``t``, the Heaviside/root vector
# ``h``, Euler's ``e`` …). A model entity that reuses one of those names fails code generation
# with a cryptic ``KeyError: 'amici_<name>'`` — which looks nothing like a name clash. Map the
# clash-prone ones to a concrete rename so the rejection is actionable.
_AMICI_RESERVED_HINTS = {
    "h": "n_hill or hill_coeff", "e": "eff or efficiency", "t": "tau or t_delay",
    "x": "x_state", "k": "k_rate", "p": "prot", "w": "w_rate",
}
_AMICI_RESERVED_RE = re.compile(r"amici_([A-Za-z]\w?)\b")


def _compile_fit_failure_message(exc: Exception) -> str:
    """Turn an AMICI/pyPESTO compile-or-fit failure into an actionable rejection message.

    A model symbol whose name collides with an AMICI-reserved codegen name (``h``, ``t``,
    ``x`` …) fails compilation with a cryptic ``'amici_<name>'`` KeyError that reads nothing
    like a name clash — the generic message below blamed an "ill-posed ODE" and sent the agent
    off to debug the maths (which is why the repressilator's first, correct model, whose Hill
    exponent was named ``h``, was abandoned). Detect the clash (a *short* symbol after the
    ``amici_`` codegen prefix — long ones like ``amici_model`` are internals, not user symbols)
    and tell it to rename; otherwise fall back to the ill-posed hint, which does describe the
    genuine non-integrable-ODE case.
    """
    m = _AMICI_RESERVED_RE.search(str(exc))
    if m:
        name = m.group(1)
        suggest = _AMICI_RESERVED_HINTS.get(name, f"{name}_rate")
        return (
            f"the model uses a symbol named '{name}', which collides with a name AMICI reserves "
            "for its generated code (reserved short symbols include h, t, x, k, p, w, e). Rename "
            f"it everywhere it appears in the Antimony (e.g. '{name}' -> '{suggest}') and "
            "re-author with set_model — the model structure itself may well be fine."
        )
    return (
        f"the model passed validation but could not be compiled/fit: {exc}. This usually means "
        "it is ill-posed for estimation — an unstable or non-integrable ODE, or a rate law the "
        "solver cannot handle. Re-check that the simulated trajectories stay bounded and the "
        "initial conditions are sane, then re-author with set_model."
    )


class OperationError(PetabMcpError):
    """An invalid model operation (bad justification, unparseable reaction, ...)."""


# -- commit pipeline -------------------------------------------------------
def _observed_symbol_ids(problem) -> set[str]:
    """Identifiers that appear in any observable formula — the 'observed' entities.

    Uses the canonical :func:`reconcile.formula_symbols` extractor (with its negative
    lookbehind) rather than a bare ``[A-Za-z_]\\w*`` scan, so a scientific-notation literal
    like ``1e5`` never leaks a bogus ``e5`` identifier — matching every other
    formula-symbol site in the package.
    """
    from petab.v1.C import OBSERVABLE_FORMULA

    ids: set[str] = set()
    obs_df = getattr(problem, "observable_df", None)
    if obs_df is not None and OBSERVABLE_FORMULA in obs_df.columns:
        for formula in obs_df[OBSERVABLE_FORMULA]:
            ids |= R.formula_symbols(formula)
    return ids


def _reaches_observed(sid: str, edges: dict[str, set[str]], targets: set[str]) -> bool:
    """Whether ``sid`` feeds — through the influence graph — any observed species in ``targets``."""
    seen: set[str] = set()
    stack: list[str] = list(edges.get(sid, ()))
    while stack:
        cur = stack.pop()
        if cur in targets:
            return True
        if cur in seen:
            continue
        seen.add(cur)
        stack.extend(edges.get(cur, ()))
    return False


def _parsimony_warnings(problem, old_species: set[str]) -> list[str]:
    """Advisory warning for an edit that adds a *genuinely disconnected* hidden species.

    Data only constrains species that connect to an observable, so a newly introduced species
    that feeds no observable — a dead-end sink like ``B -> C`` — can degenerately fit a single
    observable just as well as a repair that reuses existing species (e.g. making a reaction
    reversible). We flag *only* that case, in-loop, without rejecting the (structurally valid)
    edit.

    We deliberately do NOT flag species that are legitimately unobserved, because doing so
    trained the agent to ignore the warning entirely (it fired 62× on correct choices in one
    benchmark). Suppressed:

    * **condition-table inputs** — a species set per-condition (a dose/genotype column);
    * **held const/boundary species and rule-driven forcing terms** — model INPUTS, not free
      states (see :func:`reconcile.input_species`);
    * **any species that feeds an observed species** — an enzyme, a substrate, an upstream
      regulator reaches an observable through the reaction network, so the data does constrain
      it (see :func:`reconcile.species_influence_graph`).
    """
    sm = problem.model.sbml_model
    new_species = set(R.species_ids(sm)) - old_species
    if not new_species:
        return []
    observed = _observed_symbol_ids(problem)
    unobserved = new_species - observed
    if not unobserved:
        return []

    exempt = R.condition_column_symbols(problem) | R.input_species(sm)
    observed_species = observed & set(R.species_ids(sm))
    edges = R.species_influence_graph(sm)
    disconnected = sorted(
        sid
        for sid in unobserved
        if sid not in exempt and not _reaches_observed(sid, edges, observed_species)
    )
    if not disconnected:
        return []
    return [
        f"Added unobserved species {disconnected}: they feed no observable, come from no "
        "condition-table input, and are set by no forcing rule — genuinely disconnected hidden "
        "states that no data constrains. A repair that reuses observed/existing species (e.g. "
        "making an existing reaction reversible) is usually preferable, since a disconnected "
        "species can degenerately fit the data."
    ]


def _commit_change(
    session,
    op: str,
    op_args: dict[str, Any],
    justification: str,
    references: list[str] | None,
    *,
    model_edit: Callable[[Any], Any] | None = None,
    model_build: Callable[[], tuple[Any, Any]] | None = None,
    param_edit: Callable[[Any], Any] | None = None,
    # References are optional (only a non-empty justification is required); every caller passes
    # this explicitly, so the default just keeps a future caller from silently making them
    # mandatory again.
    require_references: bool = False,
) -> dict[str, Any]:
    """Apply one model/parameter edit to the working model — WITHOUT fitting or versioning.

    ``model_edit(sbml_model)`` edits a *clone* of the model (triggers reconcile);
    ``model_build()`` returns a brand-new ``(SbmlModel, apply_result)`` that *replaces*
    the model wholesale (used to zero-shot a model when none exists — also reconciled);
    ``param_edit(problem)`` edits the parameter table in place. A ``param_edit`` may accompany a
    model build/edit (``set_model(parameters=[...])``, issue #114): both then land in the SAME
    transaction — the parameter edit runs AFTER the reconcile, so it can target a parameter the
    re-author just introduced, and a rejected entry reverts the re-author too. On any failure the
    model and parameter table are reverted so an edit is atomic.

    Issue #81 decoupled editing from fitting: an edit mutates the working model and reconciles +
    validates it, but does NOT fit and does NOT snapshot a version — the explicit ``fit`` tool does
    both (fit + version), and the structure-level de-duplication (issue #59) moved there too. The
    edit is recorded as a *pending edit* (:meth:`session.record_edit`) and logged as an ``edit``
    provenance event; call ``fit`` to score the assembled model. Returns ``applied: true`` with the
    reconcile/validation/warnings — but no fit metrics, since no fit ran.
    """
    if model_build is not None and model_edit is not None:
        raise ValueError("_commit_change takes at most one of model_build, model_edit.")
    if model_build is None and model_edit is None and param_edit is None:
        raise ValueError(
            "_commit_change requires at least one callback: a model_build OR a model_edit "
            "(optionally with a param_edit), or a param_edit on its own."
        )
    if not isinstance(justification, str) or not justification.strip():
        raise OperationError("A non-empty justification is required for this operation.")
    resolved = resolve_references(session.resolver, references, required=require_references)

    with session.lock:
        old_model = session.problem.model
        old_sbml = old_model.to_sbml_str()
        old_species = set(R.species_ids(old_model.sbml_model))
        old_params = session.problem.parameter_df.copy(deep=True)

        try:
            apply_result: Any = None
            if model_build is not None:
                candidate, apply_result = model_build()
                session.problem.model = candidate
                recon = R.reconcile(session.problem)
            elif model_edit is not None:
                candidate = SbmlModel.from_string(old_sbml, model_id=old_model.model_id)
                apply_result = model_edit(candidate.sbml_model)
                session.problem.model = candidate
                recon = R.reconcile(session.problem)
            else:
                recon = None
            if param_edit is not None:
                # AFTER the reconcile: the parameter table only holds the re-authored model's
                # parameters once reconcile has run, so a bundled bound edit can target a symbol
                # this very (re-)author introduced — and an entry naming a parameter the re-author
                # dropped is rejected loudly instead of silently doing nothing.
                param_result = param_edit(session.problem)
                apply_result = (
                    {"model": apply_result, "parameters": param_result}
                    if recon is not None else param_result
                )
            # A parameter that ships FIXED and is a mechanistic/structural constant (a
            # conservation total, a known rate) must not be removed or freed by an edit —
            # validation rejects either. Best-effort: an unreadable v0 yields an empty set.
            try:
                protected_fixed = session.protected_fixed_parameter_ids()
            except Exception:  # noqa: BLE001 - never let the lookup break an edit
                protected_fixed = None
            report = validate_problem(session.problem, protected_fixed_ids=protected_fixed)
            if not report["valid"]:
                raise ValidationError(
                    f"Operation {op!r} rejected: the resulting problem is invalid.",
                    issues=[i["message"] for i in report["issues"]
                            if i["level"] in ("error", "critical")],
                )
        except Exception:
            # An edit is atomic: on any failure (reconcile, lint) restore the model and
            # parameter table so nothing changes. No fit runs here — that is the `fit` tool.
            session.problem.model = old_model
            session.problem.parameter_df = old_params
            raise

        new_sbml = session.problem.model.to_sbml_str()  # serialize once; reused for both hashes
        warnings = (_parsimony_warnings(session.problem, old_species) if recon is not None else [])
        param_diff = {
            "added": recon["parameters"]["added"] if recon else [],
            "removed": recon["parameters"]["removed"] if recon else [],
            "pruned_species": recon["pruned"]["species"] if recon else [],
        }
        # Per-species initial-condition account: for every species, whether its declared t=0
        # is FIXED (survives into the simulated t=0 state), ESTIMATED (the fit sets it), or
        # overridden per-condition — so the caller can see immediately whether a t=0 fix took,
        # instead of guessing IC naming schemes that all fit identically (issue #57). Computed
        # whenever a model exists (cheap, analytic); `newly_added` marks ICs this op introduced.
        initial_conditions = (
            R.initial_conditions_report(session.problem, added=set(param_diff["added"]))
            if session.has_model()
            else []
        )
        # Log the edit (no version — versions are fit snapshots, see `fit`) and mark it pending.
        event = session.provenance.append_event({
            "type": "edit",
            "operation": op,
            "args": op_args,
            "result": apply_result if isinstance(apply_result, (str, dict, list)) else None,
            "justification": justification,
            "references": resolved,
            "parameter_diff": param_diff,
            "warnings": warnings,
            "model_hash_before": sha16(old_sbml),
            "model_hash_after": sha16(new_sbml),
            # Problem hash = model + parameter table, so a parameter-only edit (bounds/scale)
            # is reflected here even though the two model hashes above are identical.
            "problem_hash_before": problem_fingerprint(old_sbml, old_params),
            "problem_hash_after": problem_fingerprint(new_sbml, session.problem.parameter_df),
        })
        session.record_edit(op, op_args, justification)
        n_pending = len(session.pending_edits)

    return {
        "applied": True,
        "operation": op,
        "version": None,
        "reconcile": recon,
        "initial_conditions": initial_conditions,
        "references": resolved,
        "justification": justification,
        "validation": report,
        "warnings": warnings,
        "parameter_diff": param_diff,
        "pending_fit": True,
        "n_pending_edits": n_pending,
        "event_timestamp": event["timestamp"],
        "message": (
            "Edit applied to the working model but NOT yet scored — no fit ran and no version was "
            "created. Make any further edits that belong to this change, then call fit() to fit the "
            "free parameters, snapshot a version, and get the fit metrics."
        ),
    }


# -- public operations: model ----------------------------------------------
def set_model(
    session, antimony: str, justification: str,
    references: list[str] | None = None,
    parameters: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Author the WHOLE model from Antimony — the single model editor.

    This BUILDS the first model when the problem ships model-less (``model_initialized:
    false``) AND re-authors/replaces an existing model wholesale. Every structural change — adding
    or removing a reaction, changing a rate law — is made by re-authoring the whole model in one
    coherent step. Like every edit it reconciles SBML↔PEtab (parametric ICs + parameter-table
    sync) and lints, reverting on failure so a bad (re-)author changes nothing — but it does NOT
    fit and does NOT create a version (issue #81); call ``fit`` to score the model. Re-authoring
    the same structure is cheap (no fit); the de-duplication that used to gate a re-author now
    gates ``fit`` (issue #59), so an identical structure is only fit once.

    Because the whole model is replaced, reconcile drops the parameter-table rows of any
    parameter the new model no longer declares. That silently discards a previously-fitted
    value, so this warns when the re-author removes a previously-ESTIMATED parameter (fixed
    noise/initial-condition parameters and anything still referenced by the observable/noise
    or condition tables are preserved). Dropping (or reintroducing as estimated) a shipped-FIXED
    mechanistic constant — a conservation total, a known rate — is a harder case: the edit
    pipeline's validation REJECTS it outright (see ``validation._fixed_constant_role_issues``).

    ``parameters`` optionally carries the SAME per-parameter entries ``change_parameters`` takes
    (issue #114). They are applied in the same transaction as the re-author, AFTER the reconcile —
    so an entry may target a parameter this author introduces, and a rejected entry reverts the
    re-author too. ``None``/``[]`` means "no parameter edit"; the whole bundle is recorded as ONE
    edit event, since it is one logical change.
    """
    if not isinstance(antimony, str) or not antimony.strip():
        raise OperationError("set_model requires a non-empty Antimony model string.")

    normalized = (
        _normalize_parameter_changes(parameters, label="set_model", arg="parameters")
        if parameters else []
    )

    # Snapshot the currently-estimated parameter ids so we can flag any the re-author drops.
    from petab.v1.C import ESTIMATE

    prior_estimated: set[str] = set()
    try:
        df = session.problem.parameter_df
        if ESTIMATE in df.columns:
            prior_estimated = {str(p) for p in df.index[estimated_mask(df)]}
    except Exception:  # never let the warning bookkeeping break the op
        prior_estimated = set()

    def build():
        try:
            candidate = SbmlModel.from_antimony(antimony, model_id=None)
        except Exception as exc:
            # A symbol reused as a stoichiometry coefficient (e.g. the SAME parameter on both sides,
            # `N A -> N B`) makes libantimony stamp two species references with the same id, which
            # collides and fails the conversion — the same #89 footgun, so give the actionable
            # guidance instead of the opaque libSBML error (see validation.stoichiometry_conflict_issues).
            conflict = stoichiometry_conflict_issues(exc)
            if conflict:
                raise ValidationError(
                    "set_model rejected: a reaction reuses a parameter/symbol as a stoichiometry "
                    "coefficient (e.g. on both sides), which Antimony cannot represent.",
                    issues=[i["message"] for i in conflict],
                ) from exc
            raise ModelUpdateError(f"Failed to parse antimony model: {exc}") from exc
        candidate.rel_path = getattr(session.problem.model, "rel_path", None) or "model.xml"
        sm = candidate.sbml_model
        # A parameter written as a reaction stoichiometry coefficient (e.g. `-> N_burst X`) is
        # frozen to a constant and dropped by the Antimony→SBML conversion; libantimony leaves the
        # dropped symbol as the SpeciesReference id, so catch it on the freshly-converted candidate
        # before it is committed (see validation.frozen_parametric_stoichiometry_issues, issue #89).
        stoich_issues = frozen_parametric_stoichiometry_issues(sm)
        if stoich_issues:
            raise ValidationError(
                "set_model rejected: a reaction uses a parameter/symbol as a stoichiometry "
                "coefficient, which Antimony freezes to a constant and drops.",
                issues=[i["message"] for i in stoich_issues],
            )
        return candidate, {
            "model_id": candidate.model_id,
            "n_species": sm.getNumSpecies(),
            "n_reactions": sm.getNumReactions(),
        }

    op_args: dict[str, Any] = {"antimony_chars": len(antimony)}
    if normalized:
        op_args["changes"] = _changes_arg(normalized)
    result = _commit_change(
        session, "set_model", op_args, justification, references,
        model_build=build,
        param_edit=_parameter_edit(session, normalized) if normalized else None,
        require_references=False,
    )

    # Guardrail: a re-author that no longer declares a previously-estimated parameter drops
    # its row (and its fitted value). Surface that — it is the one real footgun of a wholesale
    # replace; fixed/table-referenced parameters are not affected.
    removed = ((result.get("reconcile") or {}).get("parameters") or {}).get("removed") or []
    dropped = sorted(p for p in removed if str(p) in prior_estimated)
    if dropped:
        result["warnings"] = list(result.get("warnings") or []) + [
            f"set_model dropped previously-estimated parameter(s) {dropped}: their fitted "
            "values are gone. Re-declare them in the Antimony if that was unintended."
        ]
    return result


# -- public operations: parameters (justification required, references optional) --
def _require_positive_for_log(problem, parameter_id: str, scale: str) -> None:
    if scale not in ("log", "log10"):
        return
    row = problem.parameter_df.loc[parameter_id]
    for field in ("lowerBound", "nominalValue"):
        if field in problem.parameter_df.columns:
            val = row[field]
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if fval == fval and fval <= 0:  # not NaN and non-positive
                raise OperationError(
                    f"Cannot use {scale} scale for {parameter_id!r}: {field}={val} must be > 0."
                )


def _reject_sigma_bound_widening(problem, parameter_id: str, upper_bound: float) -> None:
    """Refuse to widen an ESTIMATED noise-sigma parameter's upperBound past its data-driven cap
    (see ``R.sigma_upper_bound_cap``). Without this, a fit pinned at that cap could be "fixed"
    the same way a benchmark run once fixed it — by widening the box until the sigma can
    inflate far enough to make an under-mechanistic model look statistically adequate again.
    A FIXED sigma is left alone: its bounds don't affect a fit it's never optimized over. This
    only ever REJECTS an explicit widening attempt — it never rewrites a bound itself."""
    if parameter_id not in R.noise_sigma_ids(problem):
        return
    if not R.is_parameter_estimated(problem, parameter_id):
        return
    cap = R.sigma_upper_bound_cap(problem, parameter_id)
    if cap is not None and float(upper_bound) > cap:
        raise OperationError(
            f"Cannot widen the upperBound of {parameter_id!r} to {upper_bound:.6g}: noise-sigma "
            f"parameters are capped at {cap:.6g} ({NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD:.0%} of "
            "this observable's raw measurement std) so a fit can't hide a wrong mechanism by "
            "inflating the assumed noise. If the fit is pinned at this bound, revisit the model "
            "structure — don't widen the sigma further."
        )


def _pinned_side(session, problem, parameter_id: str) -> str | None:
    """``"lower"``/``"upper"`` if the parameter's last fitted value sits AT that bound (it
    re-pinned), else ``None``. Reads the current HEAD fit from ``session.simulation`` against
    the parameter table's current bounds — the state as it is BEFORE this bounds edit applies.
    ``None`` (no block) whenever there is no fit to judge against."""
    from . import fitting

    fitted = (getattr(session, "simulation", None) or {}).get("fitted_parameters")
    if not fitted:
        return None
    for flag in fitting.parameters_at_bound(fitted, problem.parameter_df):
        if flag["parameter_id"] == parameter_id:
            return flag["side"]
    return None


def _widening_history(session, parameter_id: str, side: str) -> tuple[int, float | None]:
    """``(count, last_bic_gain)`` for the CURRENT run of widenings of ``parameter_id`` on ``side``.

    Reconstructed from each committed version's STORED parameter table (not from event args), so
    it is faithful across fits, rollbacks, and resets alike — the version snapshot is the
    ground truth of what the bounds actually were, however the version was produced. Walking the
    version history in order:

    * ``count`` increments on every step where the ``side`` bound moved OUTWARD (a widening) and
      RESETS to 0 on any step where it moved INWARD — a ``rollback_to`` / ``reset`` (or an
      explicit narrowing) restores a smaller box and legitimately starts exploration over, so the
      count reflects only the current uninterrupted run of widenings, not stale ones already undone.
    * ``last_bic_gain`` is the BIC improvement the MOST RECENT widening bought — ``delta_bic =
      2 * delta_nll``, since a bounds-only re-fit keeps k and n fixed — or ``None`` when either
      version's ``neg_log_likelihood`` is missing (e.g. a simulate-only commit).

    Needs no extra per-session state; ``None``/``0`` whenever the history can't be read."""
    prov = getattr(session, "provenance", None)
    if prov is None:
        return 0, None
    bound_col = UPPER_BOUND if side == "upper" else LOWER_BOUND
    try:
        metas = prov.list_versions()
    except Exception:  # noqa: BLE001 - unreadable version store
        return 0, None
    nllh_by_version: dict[int, float | None] = {}
    for m in metas:
        v = m.get("version")
        if v is None:
            continue
        nll = m.get("neg_log_likelihood")
        nllh_by_version[int(v)] = float(nll) if nll is not None else None

    count = 0
    last_gain: float | None = None
    prev_bound: float | None = None
    prev_nllh: float | None = None
    for v in sorted(nllh_by_version):
        try:
            bound = float(prov.get_parameter_df(v).loc[parameter_id, bound_col])
        except Exception:  # noqa: BLE001 - parameter absent at this version / unreadable snapshot
            continue
        nllh = nllh_by_version[v]
        if prev_bound is not None:
            if (bound > prev_bound) if side == "upper" else (bound < prev_bound):  # widened
                count += 1
                last_gain = (2.0 * (prev_nllh - nllh)
                             if (prev_nllh is not None and nllh is not None) else None)
            elif (bound < prev_bound) if side == "upper" else (bound > prev_bound):  # narrowed
                count, last_gain = 0, None
        prev_bound, prev_nllh = bound, nllh
    return count, last_gain


def _reject_repeated_widening(
    session, problem, parameter_id: str, lower_bound: float, upper_bound: float,
) -> None:
    """Refuse a THIRD widening of a parameter that keeps re-pinning to its bound for a negligible
    fit gain. A parameter pinned at a bound with a flat / collinear FIM direction (see
    ``get_identifiability``) is UNIDENTIFIABLE, not under-explored: each widening just lets the
    optimum slide to the new edge and re-pin, chasing the search space toward infinity for ~0 BIC
    (armistead widened one rate 4x this way before fixing it). The skill says widen at most once;
    this is the hard backstop. Noise sigmas have their own, stricter cap
    (:func:`_reject_sigma_bound_widening`) and are left to it. This only ever REJECTS — it never
    rewrites a bound — and a missing fit / provenance simply means no block."""
    if parameter_id in R.noise_sigma_ids(problem):
        return
    if not R.is_parameter_estimated(problem, parameter_id):
        return
    side = _pinned_side(session, problem, parameter_id)
    if side is None:  # not currently re-pinning — a normal, allowed bounds move
        return
    bound_col = UPPER_BOUND if side == "upper" else LOWER_BOUND
    try:
        current = float(problem.parameter_df.loc[parameter_id, bound_col])
    except (KeyError, TypeError, ValueError):
        return
    proposed = float(upper_bound if side == "upper" else lower_bound)
    widening = proposed > current if side == "upper" else proposed < current
    if not widening:  # narrowing or shifting the other side — not the pathology this guards
        return
    count, last_gain = _widening_history(session, parameter_id, side)
    if count < 2:  # a first or second widening is fine; the guard blocks the third
        return
    # Block ONLY on a KNOWN, negligible gain — the backstop is defined by negligible BIC
    # improvement, so an unknown gain (e.g. a simulate-only commit with no neg_log_likelihood)
    # must not trigger it, and neither may a widening that is still buying a real improvement.
    if last_gain is None or last_gain >= WIDENING_NEGLIGIBLE_BIC_GAIN:
        return
    raise OperationError(
        f"Refusing to widen {parameter_id!r} a {count + 1}th time: it re-pinned at its {side} "
        f"bound after {count} previous widenings for a negligible ~{last_gain:.2g} BIC gain. A "
        "parameter pinned at a bound with a flat/collinear FIM direction (check "
        "get_identifiability) is UNIDENTIFIABLE, not under-explored — widening chases it toward "
        "infinity while it keeps re-pinning. Instead re-author with set_model to REMOVE its "
        "reaction or use its LIMITING form (make the reaction irreversible / the step "
        "instantaneous); if neither is possible, LEAVE it pinned — do not widen again or "
        "hard-code a value (there is no value/estimate tool, and a literal is just a magic "
        "number — values come only from the fit)."
    )


def _normalize_parameter_changes(
    changes: list[dict[str, Any]], label: str = "change_parameters", arg: str = "changes",
) -> list[dict[str, Any]]:
    """Validate + normalize a ``changes`` list into ``{parameter_id, lower_bound, upper_bound,
    scale}`` entries (missing fields as ``None``), raising :class:`OperationError` on anything
    malformed. Shared by ``change_parameters`` and ``set_model(parameters=[...])`` so both paths
    enforce exactly the same entry contract (issue #114); ``label`` names the offending argument
    in the rejection message, so a bundled entry doesn't report a tool the caller never called."""
    if not isinstance(changes, list) or not changes:
        raise OperationError(
            f"{label}: {arg!r} must be a non-empty list of "
            "{parameter_id, lower_bound?, upper_bound?, scale?} entries."
        )
    normalized: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, ch in enumerate(changes):
        if not isinstance(ch, dict):
            raise OperationError(f"{label}: entry {i} is not an object.")
        pid = ch.get("parameter_id")
        if not pid or not isinstance(pid, str):
            raise OperationError(
                f"{label}: entry {i} is missing a string 'parameter_id'."
            )
        if pid in seen:
            raise OperationError(
                f"{label}: {pid!r} appears in more than one entry — combine its bound "
                "and scale edits into a single entry."
            )
        seen.add(pid)
        unknown = set(ch) - {"parameter_id", "lower_bound", "upper_bound", "scale"}
        if unknown:
            raise OperationError(
                f"{label}: entry for {pid!r} has unknown key(s) {sorted(unknown)} — "
                "allowed keys are parameter_id, lower_bound, upper_bound, scale (a misspelled key "
                "would otherwise be silently ignored)."
            )
        lb, ub, scale = ch.get("lower_bound"), ch.get("upper_bound"), ch.get("scale")
        if lb is None and ub is None and scale is None:
            raise OperationError(
                f"{label}: the entry for {pid!r} changes nothing — give "
                "lower_bound+upper_bound and/or scale."
            )
        if (lb is None) != (ub is None):
            raise OperationError(
                f"{label}: {pid!r} sets only one bound — a bound edit needs BOTH "
                "lower_bound and upper_bound (or neither)."
            )
        # Bounds must be real numbers HERE, not at the point of writing them: `update_parameters`
        # coerces with a bare `float(value)`, whose ValueError is not a PetabMcpError and so
        # escapes the tool's error wrapper as an unstructured MCP failure instead of a clean
        # `applied: false`. (bool is an int subclass — reject it explicitly, `True` is not a bound.)
        for field, value in (("lower_bound", lb), ("upper_bound", ub)):
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise OperationError(
                    f"{label}: {pid!r} has a non-numeric {field} ({value!r}) — bounds must be "
                    "numbers."
                )
            if value != value or value in (float("inf"), float("-inf")):  # NaN / infinity
                raise OperationError(
                    f"{label}: {pid!r} has a non-finite {field} ({value!r}) — bounds must be "
                    "finite numbers."
                )
        normalized.append(
            {"parameter_id": pid, "lower_bound": lb, "upper_bound": ub, "scale": scale}
        )
    return normalized


def _parameter_edit(session, normalized: list[dict[str, Any]]) -> Callable[[Any], dict[str, Any]]:
    """Build the ``param_edit(problem)`` callback that applies ``normalized`` (from
    :func:`_normalize_parameter_changes`) as ONE transactional write, with every per-parameter
    backstop. Shared by ``change_parameters`` and ``set_model(parameters=[...])``."""
    from .parameters import update_parameters

    def edit(problem):
        updates: list[dict[str, Any]] = []
        scale_checks: list[tuple[str, str]] = []  # (pid, scale) to verify AFTER the batched update
        changed = []
        for ch in normalized:
            pid, lb, ub, scale = ch["parameter_id"], ch["lower_bound"], ch["upper_bound"], ch["scale"]
            update: dict[str, Any] = {"id": pid}
            if ub is not None:
                # Widening backstops read the pre-edit bound, so they must run BEFORE any update.
                # Entries are distinct parameters, so checking them all against the pristine table
                # (before the single batched write below) is equivalent to interleaving.
                _reject_sigma_bound_widening(problem, pid, ub)
                _reject_repeated_widening(session, problem, pid, lb, ub)
                update["lowerBound"], update["upperBound"] = lb, ub
            if scale is not None:
                update["parameterScale"] = scale
                scale_checks.append((pid, scale))
            updates.append(update)
            summary: dict[str, Any] = {"parameter_id": pid}
            if ub is not None:
                summary["lower_bound"], summary["upper_bound"] = lb, ub
            if scale is not None:
                summary["scale"] = scale
            changed.append(summary)
        # One transactional write for the whole batch — a single parameter-table snapshot.
        update_parameters(problem, updates)
        for pid, scale in scale_checks:
            # Positive-box check reads the NEW scale + bounds, so it runs AFTER the update.
            _require_positive_for_log(problem, pid, scale)
        return {"changed": changed}

    return edit


def _changes_arg(normalized: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The normalized entries as they go into the provenance event args (``None``s dropped)."""
    return [{k: v for k, v in c.items() if v is not None} for c in normalized]


def change_parameters(
    session,
    changes: list[dict[str, Any]],
    justification: str,
    references: list[str] | None = None,
) -> dict[str, Any]:
    """Batch-edit the bounds and/or scale of one or more parameters in ONE atomic edit.

    ``changes`` is a non-empty list of per-parameter dicts, each naming ``parameter_id`` and any
    of ``lower_bound`` / ``upper_bound`` (a bound edit — give BOTH or neither) and ``scale``
    ('lin'|'log'|'log10'). Group everything that belongs to one coherent retuning into a single
    call, then ``fit`` once to score it. Every per-parameter backstop still applies (a re-pinned
    parameter's 3rd widening, a noise-sigma cap, a log scale on a non-positive box); because the
    whole batch is atomic, one rejected entry reverts ALL of them and nothing changes. An EDIT:
    mutates the parameter table but does NOT fit — the ``fit`` tool's optional ``fit=true`` bundle
    (issue #114) is wired at the server layer, not here."""
    normalized = _normalize_parameter_changes(changes)
    return _commit_change(
        session, "change_parameters", {"changes": _changes_arg(normalized)},
        justification, references,
        param_edit=_parameter_edit(session, normalized), require_references=False,
    )


# -- bundled edit + fit (issue #114) ---------------------------------------
def bundle_result(
    session, edit_result: dict[str, Any], fit_result: dict[str, Any],
    head_before: int | None = None,
) -> dict[str, Any]:
    """Merge an edit result and the fit result of a bundled ``fit=true`` call.

    The two shapes are kept SEPARATE rather than flattened: the whole fit result (its metrics, or
    its ``error`` / ``no_change`` short-circuit) lands untouched under ``fit``, so no key of either
    result is clobbered — in particular the edit's own ``warnings`` (e.g. "the re-author dropped a
    previously-estimated parameter") survive alongside the fit's. Only ``version``/``committed``
    are mirrored to the top level, because "which version did this call produce?" is the first
    thing a caller reads.

    On a failure the message does NOT prescribe a fix, it forwards ``fit.error`` — because that
    error is not always "the model is broken". The identical-structure loop guard reaches the same
    branch, and its message says the opposite (stop re-fitting this structure, go back with
    ``restore_version(n)``); a top-level "repair the model and call fit() again" would send the
    agent straight back into the loop the guard exists to break, and the agent reads the message
    first. Any future rejection reason is forwarded correctly for the same reason.

    Whether a version was created is decided by comparing HEAD against ``head_before``, NOT by
    ``has_unfit_edits()``. They usually agree, but ``session.fit`` clears the pending edits several
    statements after it snapshots — an event-log write failure in that window leaves the edits
    pending with HEAD already advanced, and the proxy would then claim nothing was scored.
    ``pending_fit`` is still reported from the session: it is a separate fact (are there edits the
    next fit would score?) and stays accurate in that window too.
    """
    committed = bool(fit_result.get("committed"))
    pending = session.has_unfit_edits()
    version = fit_result.get("version")
    head_now = session.provenance.current_version
    if committed:
        message = (
            f"Edit applied and scored: version {version} was created. "
            "Read the `fit` block for the metrics."
        )
    elif fit_result.get("no_change"):
        message = (
            "Edit applied, but the fit was a no-op: the resulting model is identical to version "
            f"{fit_result.get('duplicate_of_version')}, so its stored metrics stand (see `fit`). "
            "Make a genuinely different change, or restore_version(n) to go back."
        )
    elif head_before is not None and head_now != head_before:
        # The fit broke only AFTER snapshotting, so HEAD moved: report where it landed rather than
        # asking for a re-fit that would be a de-dup no-op against the version just created.
        version = version if version is not None else head_now
        message = (
            "Edit applied and a version was snapshotted, but the fit did not finish cleanly, so "
            f"this call carries no metrics: the working model is now version {version}. Read "
            "`fit.error`, then read that version's score with list_versions before making "
            "another change — do NOT re-fit, the model is already scored."
        )
    else:
        message = (
            "Edit APPLIED to the working model, but it was NOT scored — the edit is still in "
            "place and no version was created. The fit reported: "
            f"{fit_result.get('error') or 'no error message'} — ACT ON THAT rather than assuming "
            "the model is broken: it says whether to repair and re-fit, or to stop re-fitting "
            "this structure and go back with restore_version(n)."
        )
    return {
        **edit_result,
        "committed": committed,
        "version": version,
        "pending_fit": pending,
        "n_pending_edits": len(session.pending_edits),
        "warnings": list(edit_result.get("warnings") or [])
        + list(fit_result.get("warnings") or []),
        "fit": fit_result,
        "message": message,
    }
