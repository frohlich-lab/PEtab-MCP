"""The three constrained, justified model operations.

Every operation:

1. requires a non-empty ``justification`` (``references`` are optional; any given are
   validated via the session's :class:`ReferenceResolver`);
2. applies a structural edit to a *copy* of the model (libsbml);
3. reconciles SBML↔PEtab (parametric ICs, parameter-table sync, orphan pruning);
4. lints the candidate — on any failure the edit is reverted;
5. fits the parameters and simulates so the version records fitted values, fit metrics,
   and the simulation dataframe — a fit/simulation failure reverts the edit (atomic commit);
6. on success commits, snapshots a single new version, and logs a provenance ``alteration``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

import libsbml
from petab.v1.models.sbml_model import SbmlModel

from . import reconcile as R
from .errors import ModelUpdateError, PetabMcpError, UnknownEntityError, ValidationError
from .parameters import problem_fingerprint
from .references import resolve_references
from .validation import validate_problem

_ARROW_RE = re.compile(r"\s*(->|=>)\s*")

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


# -- reaction-line parsing -------------------------------------------------
def _parse_side(side: str) -> list[tuple[str, float]]:
    side = side.strip()
    if not side:
        return []
    terms = []
    for raw in side.split("+"):
        tok = raw.strip()
        if not tok:
            continue
        parts = tok.split()
        if len(parts) == 1:
            terms.append((parts[0], 1.0))
        elif len(parts) == 2:
            try:
                terms.append((parts[1], float(parts[0])))
            except ValueError as exc:
                raise OperationError(f"Bad stoichiometry term {tok!r}.") from exc
        else:
            raise OperationError(f"Bad stoichiometry term {tok!r}.")
    return terms


def parse_reaction_line(line: str) -> dict[str, Any]:
    """Parse ``[id:] reactants (->|=>) products [; rate_law]``."""
    text = line.strip().rstrip(";").strip()
    rate_law = None
    if ";" in text:
        text, rate_law = (p.strip() for p in text.split(";", 1))
    rid = None
    # an "id:" prefix (colon before the arrow, not part of the arrow)
    if ":" in text.split("->")[0].split("=>")[0]:
        rid, text = (p.strip() for p in text.split(":", 1))
    m = _ARROW_RE.search(text)
    if not m:
        raise OperationError(
            f"Reaction {line!r} must contain '->' or '=>' between reactants and products."
        )
    reactants = _parse_side(text[: m.start()])
    products = _parse_side(text[m.end():])
    if not reactants and not products:
        raise OperationError(f"Reaction {line!r} has no reactants or products.")
    return {"id": rid, "reactants": reactants, "products": products, "rate_law": rate_law}


# -- structural edits (applied to a candidate libsbml model) ---------------
def _ensure_species(sm, sid: str) -> None:
    if sm.getSpecies(sid) is None:
        R.add_species_with_parametric_ic(sm, sid, 0.0)


def _add_reaction_to_model(sm, spec: dict[str, Any]) -> str:
    rid = spec["id"]
    if rid is None:
        n = sm.getNumReactions() + 1
        while sm.getReaction(f"J{n}") is not None:
            n += 1
        rid = f"J{n}"
    if sm.getReaction(rid) is not None:
        raise OperationError(f"Reaction id {rid!r} already exists.")
    if not spec["rate_law"]:
        raise OperationError("A new reaction requires a rate law (after ';').")

    for sid, _ in spec["reactants"] + spec["products"]:
        _ensure_species(sm, sid)
    # new rate-law symbols that are not species become parameters
    known_species = R.species_ids(sm)
    for sym in R.rate_law_symbols(spec["rate_law"]):
        if sym not in known_species and sm.getParameter(sym) is None:
            R.add_parameter(sm, sym)

    rxn = sm.createReaction()
    rxn.setId(rid)
    rxn.setReversible(False)
    rxn.setFast(False)
    for sid, st in spec["reactants"]:
        ref = rxn.createReactant()
        ref.setSpecies(sid)
        ref.setStoichiometry(st)
        ref.setConstant(True)
    for sid, st in spec["products"]:
        ref = rxn.createProduct()
        ref.setSpecies(sid)
        ref.setStoichiometry(st)
        ref.setConstant(True)
    kl = rxn.createKineticLaw()
    math = libsbml.parseL3Formula(spec["rate_law"])
    if math is None:
        raise OperationError(f"Could not parse rate law {spec['rate_law']!r}.")
    kl.setMath(math)
    return rid


def _change_rate_law_in_model(sm, reaction_id: str, rate_law: str) -> None:
    rxn = sm.getReaction(reaction_id)
    if rxn is None:
        raise UnknownEntityError(f"No reaction with id {reaction_id!r}.")
    known_species = R.species_ids(sm)
    for sym in R.rate_law_symbols(rate_law):
        if sym not in known_species and sm.getParameter(sym) is None:
            R.add_parameter(sm, sym)
    math = libsbml.parseL3Formula(rate_law)
    if math is None:
        raise OperationError(f"Could not parse rate law {rate_law!r}.")
    kl = rxn.getKineticLaw() or rxn.createKineticLaw()
    kl.setMath(math)


def _remove_reaction_from_model(sm, reaction_id: str) -> None:
    if sm.getReaction(reaction_id) is None:
        raise UnknownEntityError(f"No reaction with id {reaction_id!r}.")
    sm.removeReaction(reaction_id)


# -- commit pipeline -------------------------------------------------------
def _model_hash(sbml_str: str) -> str:
    return hashlib.sha256(sbml_str.encode("utf-8")).hexdigest()[:16]


def _observed_symbol_ids(problem) -> set[str]:
    """Identifiers that appear in any observable formula — the 'observed' entities."""
    from petab.v1.C import OBSERVABLE_FORMULA

    ids: set[str] = set()
    obs_df = getattr(problem, "observable_df", None)
    if obs_df is not None and OBSERVABLE_FORMULA in obs_df.columns:
        for formula in obs_df[OBSERVABLE_FORMULA]:
            ids |= set(re.findall(r"[A-Za-z_]\w*", str(formula)))
    return ids


def _parsimony_warnings(problem, old_species: set[str]) -> list[str]:
    """Advisory warnings for an edit that adds species appearing in no observable.

    Data only constrains observed species, so a newly introduced *unobserved* species
    (e.g. a sink ``B -> C``) can degenerately fit a single observable just as well as a
    repair that reuses existing species (e.g. making a reaction reversible). We flag it so
    the caller sees this in-loop, without rejecting the (structurally valid) edit.
    """
    new_species = set(R.species_ids(problem.model.sbml_model)) - old_species
    unobserved = sorted(new_species - _observed_symbol_ids(problem))
    if not unobserved:
        return []
    return [
        f"Added unobserved species {unobserved}: they appear in no observable formula. "
        "A repair that reuses observed/existing species (e.g. making an existing reaction "
        "reversible) is usually preferable — an unobserved sink can degenerately fit the data."
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
    require_references: bool = True,
    n_starts: int | None = None,
) -> dict[str, Any]:
    """Shared commit pipeline for model and parameter operations.

    ``model_edit(sbml_model)`` edits a *clone* of the model (triggers reconcile);
    ``model_build()`` returns a brand-new ``(SbmlModel, apply_result)`` that *replaces*
    the model wholesale (used to zero-shot a model when none exists — also reconciled);
    ``param_edit(problem)`` edits the parameter table in place. On any failure the model
    and parameter table are reverted. When a batch is open the change is applied and
    logged but no version is snapshotted until ``commit_batch``.

    On a non-batch commit the parameters are fitted and the model simulated *before* the
    snapshot, so the single version records the fitted values, the fit metrics, and the
    simulation dataframe. ``n_starts`` overrides the server default (0 = simulate-only).
    The fit is part of the commit: a fit/simulation failure reverts the structural edit so
    the commit is atomic.
    """
    if sum(cb is not None for cb in (model_build, model_edit, param_edit)) != 1:
        raise ValueError(
            "_commit_change requires exactly one of model_build, model_edit, param_edit."
        )
    if not isinstance(justification, str) or not justification.strip():
        raise OperationError("A non-empty justification is required for this operation.")
    resolved = resolve_references(session.resolver, references, required=require_references)

    with session.lock:
        old_model = session.problem.model
        old_sbml = old_model.to_sbml_str()
        old_species = set(R.species_ids(old_model.sbml_model))
        old_params = session.problem.parameter_df.copy(deep=True)
        fitsim: dict[str, Any] = {}
        improvement: dict[str, Any] = {}

        try:
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
                apply_result = param_edit(session.problem)
                recon = None
            report = validate_problem(session.problem)
            if not report["valid"]:
                raise ValidationError(
                    f"Operation {op!r} rejected: the resulting problem is invalid.",
                    issues=[i["message"] for i in report["issues"]
                            if i["level"] in ("error", "critical")],
                )
            # Fit + simulate the validated candidate before snapshotting (non-batch
            # commits only). A fit/simulation failure falls through to the revert below —
            # a commit is atomic: it either fully succeeds or changes nothing. Wrap an
            # AMICI/pyPESTO failure (e.g. a model that lints clean but cannot be compiled —
            # a reserved-name codegen clash, an unsupported construct) as a ModelUpdateError
            # so the tool returns a clean committed=false with the message, instead of the
            # raw exception crashing the call.
            staged = session.batch_active
            if not staged:
                # A model with no estimated parameters has nothing to fit; catch it here with a
                # clear, actionable message rather than letting pyPESTO fail cryptically
                # ("'NoneType' object is not iterable") deep in the optimizer.
                if int(getattr(session.problem, "n_estimated", 0) or 0) == 0:
                    raise ModelUpdateError(
                        "the model has NO estimated (free) parameters, so there is nothing to "
                        "fit. Give it at least one free parameter — a new symbol you use in a "
                        "rate law becomes an estimated parameter automatically, so declare one "
                        "(or keep using the shipped free parameters) — then re-author with "
                        "set_model."
                    )
                try:
                    fitsim, improvement = session._auto_fit_and_improvement(n_starts)
                except PetabMcpError:
                    raise
                except Exception as exc:  # noqa: BLE001 - surface as a clean, actionable rejection
                    raise ModelUpdateError(_compile_fit_failure_message(exc)) from exc
        except Exception:
            session.problem.model = old_model
            session.problem.parameter_df = old_params
            raise

        new_ant = session.problem.model.to_antimony()
        warnings = _parsimony_warnings(session.problem, old_species) if recon is not None else []
        param_diff = {
            "added": recon["parameters"]["added"] if recon else [],
            "removed": recon["parameters"]["removed"] if recon else [],
            "pruned_species": recon["pruned"]["species"] if recon else [],
        }
        event_base = {
            "type": "alteration",
            "operation": op,
            "args": op_args,
            "result": apply_result if isinstance(apply_result, (str, dict, list)) else None,
            "justification": justification,
            "references": resolved,
            "parameter_diff": param_diff,
            "warnings": warnings,
            "model_hash_before": _model_hash(old_sbml),
            "model_hash_after": _model_hash(session.problem.model.to_sbml_str()),
            # Problem hash = model + parameter table, so a parameter-only edit (bounds/scale)
            # is reflected here even though the two model hashes above are identical.
            "problem_hash_before": problem_fingerprint(old_sbml, old_params),
            "problem_hash_after": problem_fingerprint(
                session.problem.model.to_sbml_str(), session.problem.parameter_df
            ),
        }

        if staged:
            event = session.provenance.append_event({**event_base, "staged": True, "version": None})
            session.staged.append({"operation": op, "args": op_args, "justification": justification})
            version = None
        else:
            from . import fitting

            version_before = session.provenance.current_version
            nllh = fitsim.get("neg_log_likelihood")
            version = session.provenance.snapshot_version(
                model_ant=new_ant,
                parameters=session.problem.parameter_df,  # the fitted table (df)
                label=op,
                kind="alteration",
                meta={"operation": op, "args": op_args, "fit_status": fitsim.get("status"),
                      "neg_log_likelihood": nllh, "chi2": fitsim.get("chi2"),
                      "fitted_parameters": fitsim.get("fitted_parameters"), **improvement},
                simulation=fitting.build_sim_payload_for_version(fitsim),
                optimization=fitting.build_optimization_payload_for_version(session.optimization),
            )
            fitsim["version"] = version
            event = session.provenance.append_event(
                {**event_base, "version_before": version_before, "version": version,
                 "fit_status": fitsim.get("status"), "neg_log_likelihood": nllh,
                 "chi2": fitsim.get("chi2"), "fitted_parameters": fitsim.get("fitted_parameters"),
                 **improvement}
            )

    result = {
        "committed": True,
        "operation": op,
        "version": version,
        "staged": staged,
        "reconcile": recon,
        "references": resolved,
        "justification": justification,
        "validation": report,
        "warnings": warnings,
        "event_timestamp": event["timestamp"],
    }
    if not staged:
        from . import fitting

        result.update(fitting.fit_result_block(fitsim, improvement))
    return result


# -- public operations: model ----------------------------------------------
def set_model(
    session, antimony: str, justification: str,
    references: list[str] | None = None, n_starts: int | None = None,
) -> dict[str, Any]:
    """Author the WHOLE model from Antimony — the single whole-model operation.

    This BUILDS the first model when the problem ships model-less (``model_initialized:
    false``) AND re-authors/replaces an existing model wholesale; it is never refused. Use it
    to set model STRUCTURE — a coupled/symmetric rate-law change, adding or removing several
    reactions at once — in one coherent step, instead of a sequence of incremental edits that
    leave intermediate models inconsistent (the incremental ``add_reaction`` /
    ``change_rate_law`` / ``remove_reaction`` remain for local refinement). Goes through the
    same commit pipeline as every model op: SBML↔PEtab reconcile (parametric ICs +
    parameter-table sync), petab lint, and a fit on commit; any failure reverts, so a bad
    (re-)author changes nothing.

    Because the whole model is replaced, reconcile drops the parameter-table rows of any
    parameter the new model no longer declares. That silently discards a previously-fitted
    value, so this warns when the re-author removes a previously-ESTIMATED parameter (fixed
    noise/initial-condition parameters and anything still referenced by the observable/noise
    or condition tables are preserved).
    """
    if not isinstance(antimony, str) or not antimony.strip():
        raise OperationError("set_model requires a non-empty Antimony model string.")

    # Snapshot the currently-estimated parameter ids so we can flag any the re-author drops.
    from petab.v1.C import ESTIMATE

    prior_estimated: set[str] = set()
    try:
        df = session.problem.parameter_df
        if ESTIMATE in df.columns:
            prior_estimated = {str(p) for p in df.index[df[ESTIMATE].fillna(0).astype(float) != 0]}
    except Exception:  # never let the warning bookkeeping break the op
        prior_estimated = set()

    def build():
        try:
            candidate = SbmlModel.from_antimony(antimony, model_id=None)
        except Exception as exc:
            raise ModelUpdateError(f"Failed to parse antimony model: {exc}") from exc
        candidate.rel_path = getattr(session.problem.model, "rel_path", None) or "model.xml"
        sm = candidate.sbml_model
        return candidate, {
            "model_id": candidate.model_id,
            "n_species": sm.getNumSpecies(),
            "n_reactions": sm.getNumReactions(),
        }

    result = _commit_change(
        session, "set_model", {"antimony_chars": len(antimony)}, justification, references,
        model_build=build, require_references=False, n_starts=n_starts,
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


def _require_model(session, op: str) -> None:
    """Structural edits operate on an existing model; there must be one to edit."""
    if not session.has_model():
        raise OperationError(
            f"Cannot {op}: no model has been initialized yet. Author one first with "
            "set_model (Antimony)."
        )


def add_reaction(
    session, reaction: str, justification: str, references: list[str] | None = None,
    n_starts: int | None = None,
) -> dict[str, Any]:
    _require_model(session, "add a reaction")
    spec = parse_reaction_line(reaction)
    return _commit_change(
        session, "add_reaction", {"reaction": reaction}, justification, references,
        model_edit=lambda sm: {"reaction_id": _add_reaction_to_model(sm, spec)},
        require_references=False, n_starts=n_starts,
    )


def change_rate_law(
    session, reaction_id: str, rate_law: str, justification: str,
    references: list[str] | None = None, n_starts: int | None = None,
) -> dict[str, Any]:
    _require_model(session, "change a rate law")

    def edit(sm):
        _change_rate_law_in_model(sm, reaction_id, rate_law)
        return {"reaction_id": reaction_id}

    return _commit_change(
        session, "change_rate_law", {"reaction_id": reaction_id, "rate_law": rate_law},
        justification, references, model_edit=edit, require_references=False, n_starts=n_starts,
    )


def remove_reaction(
    session, reaction_id: str, justification: str, references: list[str] | None = None,
    n_starts: int | None = None,
) -> dict[str, Any]:
    _require_model(session, "remove a reaction")

    def edit(sm):
        _remove_reaction_from_model(sm, reaction_id)
        return {"reaction_id": reaction_id}

    return _commit_change(
        session, "remove_reaction", {"reaction_id": reaction_id}, justification, references,
        model_edit=edit, require_references=False, n_starts=n_starts,
    )


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


def change_parameter_scale(
    session, parameter_id: str, scale: str, justification: str,
    references: list[str] | None = None, n_starts: int | None = None,
) -> dict[str, Any]:
    from .parameters import update_parameters

    def edit(problem):
        update_parameters(problem, [{"id": parameter_id, "parameterScale": scale}])
        _require_positive_for_log(problem, parameter_id, scale)
        return {"parameter_id": parameter_id, "scale": scale}

    return _commit_change(
        session, "change_parameter_scale", {"parameter_id": parameter_id, "scale": scale},
        justification, references, param_edit=edit, require_references=False, n_starts=n_starts,
    )


def change_parameter_bounds(
    session,
    parameter_id: str,
    lower_bound: float,
    upper_bound: float,
    justification: str,
    references: list[str] | None = None,
    n_starts: int | None = None,
) -> dict[str, Any]:
    from .parameters import update_parameters

    def edit(problem):
        update_parameters(
            problem,
            [{"id": parameter_id, "lowerBound": lower_bound, "upperBound": upper_bound}],
        )
        return {"parameter_id": parameter_id, "lowerBound": lower_bound, "upperBound": upper_bound}

    return _commit_change(
        session, "change_parameter_bounds",
        {"parameter_id": parameter_id, "lowerBound": lower_bound, "upperBound": upper_bound},
        justification, references, param_edit=edit, require_references=False, n_starts=n_starts,
    )
