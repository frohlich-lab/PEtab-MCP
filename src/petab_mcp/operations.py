"""The three constrained, justified model operations.

Every operation:

1. requires a non-empty ``justification`` and ≥1 ``references`` (validated via the
   session's :class:`ReferenceResolver`);
2. applies a structural edit to a *copy* of the model (libsbml);
3. reconciles SBML↔PEtab (parametric ICs, parameter-table sync, orphan pruning);
4. lints the candidate — on any failure the edit is reverted;
5. on success commits, snapshots a new version, and logs a provenance ``alteration``.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable

import libsbml
from petab.v1.models.sbml_model import SbmlModel

from . import reconcile as R
from .errors import ModelUpdateError, PetabMcpError, UnknownEntityError, ValidationError
from .parameters import parameters_to_records
from .references import resolve_references
from .validation import validate_problem

_ARROW_RE = re.compile(r"\s*(->|=>)\s*")


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


def _commit_change(
    session,
    op: str,
    op_args: dict[str, Any],
    justification: str,
    references: list[str] | None,
    *,
    model_edit: Callable[[Any], Any] | None = None,
    param_edit: Callable[[Any], Any] | None = None,
    require_references: bool = True,
) -> dict[str, Any]:
    """Shared commit pipeline for model and parameter operations.

    ``model_edit(sbml_model)`` edits a *clone* of the model (triggers reconcile);
    ``param_edit(problem)`` edits the parameter table in place. On any failure the model
    and parameter table are reverted. When a batch is open the change is applied and
    logged but no version is snapshotted until ``commit_batch``.
    """
    if not isinstance(justification, str) or not justification.strip():
        raise OperationError("A non-empty justification is required for this operation.")
    resolved = resolve_references(session.resolver, references, required=require_references)

    with session.lock:
        old_model = session.problem.model
        old_sbml = old_model.to_sbml_str()
        old_params = session.problem.parameter_df.copy(deep=True)

        try:
            if model_edit is not None:
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
        except Exception:
            session.problem.model = old_model
            session.problem.parameter_df = old_params
            raise

        new_ant = session.problem.model.to_antimony()
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
            "model_hash_before": _model_hash(old_sbml),
            "model_hash_after": _model_hash(session.problem.model.to_sbml_str()),
        }

        staged = session.batch_active
        if staged:
            event = session.provenance.append_event({**event_base, "staged": True, "version": None})
            session.staged.append({"operation": op, "args": op_args, "justification": justification})
            version = None
        else:
            version_before = session.provenance.current_version
            version = session.provenance.snapshot_version(
                model_ant=new_ant,
                parameters=parameters_to_records(session.problem),
                label=op,
                kind="alteration",
                meta={"operation": op, "args": op_args},
            )
            event = session.provenance.append_event(
                {**event_base, "version_before": version_before, "version": version}
            )

    return {
        "committed": True,
        "operation": op,
        "version": version,
        "staged": staged,
        "reconcile": recon,
        "references": resolved,
        "justification": justification,
        "validation": report,
        "event_timestamp": event["timestamp"],
    }


# -- public operations: model ----------------------------------------------
def add_reaction(session, reaction: str, justification: str, references: list[str]) -> dict[str, Any]:
    spec = parse_reaction_line(reaction)
    return _commit_change(
        session, "add_reaction", {"reaction": reaction}, justification, references,
        model_edit=lambda sm: {"reaction_id": _add_reaction_to_model(sm, spec)},
    )


def change_rate_law(
    session, reaction_id: str, rate_law: str, justification: str, references: list[str]
) -> dict[str, Any]:
    def edit(sm):
        _change_rate_law_in_model(sm, reaction_id, rate_law)
        return {"reaction_id": reaction_id}

    return _commit_change(
        session, "change_rate_law", {"reaction_id": reaction_id, "rate_law": rate_law},
        justification, references, model_edit=edit,
    )


def remove_reaction(session, reaction_id: str, justification: str, references: list[str]) -> dict[str, Any]:
    def edit(sm):
        _remove_reaction_from_model(sm, reaction_id)
        return {"reaction_id": reaction_id}

    return _commit_change(
        session, "remove_reaction", {"reaction_id": reaction_id}, justification, references,
        model_edit=edit,
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
    session, parameter_id: str, scale: str, justification: str, references: list[str] | None = None
) -> dict[str, Any]:
    from .parameters import update_parameters

    def edit(problem):
        update_parameters(problem, [{"id": parameter_id, "parameterScale": scale}])
        _require_positive_for_log(problem, parameter_id, scale)
        return {"parameter_id": parameter_id, "scale": scale}

    return _commit_change(
        session, "change_parameter_scale", {"parameter_id": parameter_id, "scale": scale},
        justification, references, param_edit=edit, require_references=False,
    )


def change_parameter_bounds(
    session,
    parameter_id: str,
    lower_bound: float,
    upper_bound: float,
    justification: str,
    references: list[str] | None = None,
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
        justification, references, param_edit=edit, require_references=False,
    )
