"""The constrained, justified model/parameter edits.

The two edit paths behind the server's single ``make_changes`` tool: :func:`set_model` authors
the whole model, :func:`change_parameters` adjusts a parameter's search space. They stay
separate functions because they are separate transactions with separate validation; the tool
layer picks one per call and then scores the result (see :meth:`session.PetabSession.fit`).

Every edit:

1. requires a non-empty ``justification`` (``references`` are optional; any given are
   validated via the session's :class:`ReferenceResolver`);
2. applies a structural edit to a *copy* of the model (libsbml);
3. reconciles SBML↔PEtab (parametric ICs, parameter-table sync, orphan pruning);
4. lints the candidate — on any failure the edit is reverted (atomic);
5. logs a provenance ``edit`` event.

``make_changes`` always runs an edit and then the fit, so an unscored edit is only ever a
transient state inside one call.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Callable

from petab.v1.C import LOWER_BOUND, UPPER_BOUND
from petab.v1.models.sbml_model import SbmlModel

from . import messages as M
from . import reconcile as R
from .constants import NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD, WIDENING_NEGLIGIBLE_BIC_GAIN
from .errors import ModelUpdateError, PetabMcpError, ValidationError
from .parameters import estimated_mask
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
# NOTE on `t`: renaming is the WRONG advice when the colliding symbol is standing in for time,
# which it almost always is. An agent told to rename writes `tau`, `tau` has nothing binding it to
# simulation time, reconcile mints it as an ESTIMATED parameter, and every `piecewise(level,
# tau >= X_time, 0)` gate silently becomes a comparison against a fitted CONSTANT — all inputs go
# time-invariant while the model still compiles, fits and commits. Measured on the 2026-08-17
# isensee sweep: 4 of 16 runs hit the collision, and the one that followed the "rename to tau"
# hint lost ~512 nll to it, recovering the whole amount in one edit once it switched to `time`.
# So `t` maps to an INSTRUCTION, not a replacement name.
_AMICI_RESERVED_HINTS = {
    "h": "n_hill or hill_coeff", "e": "eff or efficiency",
    "x": "x_state", "k": "k_rate", "p": "prot", "w": "w_rate",
}
_AMICI_RESERVED_RE = re.compile(r"amici_([A-Za-z]\w?)\b")

#: Signatures of a BUILD-environment failure rather than a modelling one. These are BUILD-CONTEXT
#: tokens only. The first version of this matched `No such file or directory` and `Permission
#: denied` anywhere in the text, which caught fit-time OSErrors (a pruned scratch dir, a missing
#: pyPESTO results.h5) and told the agent not to re-author a model that WAS the problem.
_TOOLCHAIN_RE = re.compile(
    r"fatal error:|"                          # a compiler hard-stop, e.g. a missing <span>
    r"Failed building the model extension|"
    r"build_ext|setup\.py|"                   # the OUTER exception AMICI actually re-raises
    r"\bcmake\b|\bninja\b|\bswig\b|"
    # Signatures of a compiler/host problem, NOT of bad generated code. Kept specific on purpose:
    # a model-caused codegen error ("'foo' was not declared") must stay on the ill-posed path, so
    # a bare `error:` is deliberately absent.
    r"unrecognized command line option|command not found|"
    r"No CMAKE_[A-Z_]+_COMPILER could be found|is not a full path to an existing compiler tool",
    re.IGNORECASE,
)


def _build_output_tail(exc: BaseException, max_lines: int = 25) -> str:
    """The compiler/cmake diagnostic a failed build actually produced, or ``""``.

    AMICI runs the build with ``stdout=PIPE, stderr=STDOUT`` and, on failure, prints ``e.output``
    and re-raises. So ``str(exc)`` is only ever the useless wrapper
    (``Command '[...]' returned non-zero exit status 1.``) while the REAL cause — the missing
    header, the link error, the cmake refusal — sits on the exception object and never reached
    anyone reading the rejection. Losing it cost several rounds of diagnosing a CI failure from
    the wrapper alone; surfacing the tail makes a build failure self-diagnosing wherever the
    message lands (agent, result.json, test log).

    The TAIL, because a compiler dumps pages of progress before the error that stopped it.
    """
    for candidate in (exc, exc.__cause__, exc.__context__):
        output = getattr(candidate, "output", None) or getattr(candidate, "stderr", None)
        if not output:
            continue
        if isinstance(output, bytes):
            output = output.decode("utf-8", "replace")
        lines = [ln for ln in str(output).splitlines() if ln.strip()]
        if lines:
            return "\n".join(lines[-max_lines:])
    return ""


def _is_toolchain_failure(exc: Exception) -> bool:
    """Whether a compile/fit failure is the HOST's fault rather than the model's.

    Two signals, because the text alone is not enough. AMICI catches the inner ``cmake`` error,
    PRINTS the compiler diagnostics, and re-raises the OUTER ``CalledProcessError`` — whose text is
    just ``Command '[<py>, '.../setup.py', 'build_ext', ...]' returned non-zero exit status 1.``
    with no compiler wording at all. Matching prose alone therefore missed the very failure this
    was written for (a GCC 8.5 `<span>` build), which is why the TYPE is checked first: a build is
    the only thing here that shells out, so a CalledProcessError IS a build failure.
    """
    import subprocess

    if isinstance(exc, subprocess.CalledProcessError):
        return True
    return bool(_TOOLCHAIN_RE.search(str(exc)))


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
        if name == "t":
            # NOT a rename. `t` collides because the caller is expressing TIME, and the fix is the
            # built-in symbol, not another name of its own: a free symbol has nothing binding it
            # to simulation time, so reconcile mints it as an ESTIMATED parameter and every
            # `piecewise(level, tau >= X_time, 0)` gate becomes a comparison against a fitted
            # CONSTANT — the inputs go time-invariant while the model still compiles and fits.
            # Measured on the 2026-08-17 isensee sweep: 4 of 16 runs hit this, and the one that
            # took the old "rename to tau" advice lost ~512 nll, recovering it all in one edit
            # once it switched to `time`.
            return (
                "the model uses a symbol named 't', which collides with a name AMICI reserves for "
                "its generated code. If 't' stands for TIME, do NOT rename it: write the built-in "
                "symbol `time` instead (e.g. `X := X_level * piecewise(0, time < X_start, 1)`). "
                "Any other name (`tau`, `t_now`) is a free parameter the fit will estimate, which silently makes "
                "your time-dependent inputs CONSTANT while the model still compiles. If 't' was "
                "something else entirely, rename it to a descriptive name. Re-author with "
                "make_changes — the model structure itself may well be fine."
            )
        suggest = _AMICI_RESERVED_HINTS.get(name, f"{name}_rate")
        return (
            f"the model uses a symbol named '{name}', which collides with a name AMICI reserves "
            "for its generated code (reserved short symbols include h, t, x, k, p, w, e). Rename "
            f"it everywhere it appears in the Antimony (e.g. '{name}' -> '{suggest}') and "
            "re-author with make_changes — the model structure itself may well be fine."
        )
    tail = _build_output_tail(exc)
    if _is_toolchain_failure(exc) or (tail and _TOOLCHAIN_RE.search(tail)):
        # A BUILD failure, not a modelling one. Blaming the model here is actively harmful: the
        # agent's only lever is make_changes, so it rewrites a correct model and the real cause —
        # a compiler, cmake, swig or disk problem on the host — never gets looked at. The C++20
        # case cost exactly that (a missing `<span>` from GCC 8.5 was read as an ill-posed ODE,
        # and later as "pre-existing environment breakage", before anyone read the cmake log).
        return (
            f"the model could not be BUILT, which is a toolchain problem on this host rather "
            f"than a problem with the model: {exc}. Do not re-author the model — it passed "
            "validation. Check the build environment: a C++20-capable compiler (AMICI's "
            "generated code includes <span>, so GCC >= 10 / a conda compiler is required, and "
            "CC/CXX must BOTH point at it), cmake/ninja/swig on PATH, and free space plus write "
            "permission on the build directory."
            + (f"\n\nWhat the build actually reported:\n{tail}" if tail else "")
        )
    return (
        f"the model passed validation but could not be compiled/fit: {exc}. This usually means "
        "it is ill-posed for estimation — an unstable or non-integrable ODE, or a rate law the "
        "solver cannot handle. Re-check that the simulated trajectories stay bounded and the "
        "initial conditions are sane, then re-author with make_changes."
        + (f"\n\nWhat the build actually reported:\n{tail}" if tail else "")
    )


class OperationError(PetabMcpError):
    """An invalid model operation (bad justification, unparseable reaction, ...).

    ``cause`` is an optional MACHINE-READABLE label for the rejection class, carried through to
    ``result.json``'s ``edit_rejects.by_cause``. It exists because classifying rejections by
    substring-matching their English text put three separate buckets on an unreachable branch in
    one day — a needle that also appears in an earlier entry silently steals every match, and the
    prose is free to change under the classifier. When a raiser sets this, the classifier uses it
    verbatim and the wording is nobody's contract."""

    def __init__(self, *args, cause: str | None = None):
        super().__init__(*args)
        self.cause = cause


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


_REJECTION_ISSUES_SHOWN = 3
_REJECTION_ISSUE_CHARS = 320


def _rejection_summary(issues: list[str]) -> str:
    """The rejection message, with the reasons IN it rather than only beside it.

    `issues` already travelled next to the error, but the bare
    "the resulting problem is invalid." carried no reason, and that class recovered on the next
    call only 60.4% of the time against 83-100% for rejections whose message names the fix (Aug
    8-10 sweep, 197 calls). Folding the leading reasons into the text the agent reads first is one
    concatenation and needs no new machinery.

    Truncated per issue and capped in count: libSBML emits multi-paragraph messages, and a wall of
    them costs the context it is meant to save. The full list stays on `issues`, and the count of
    what was elided is stated so a truncated view never looks complete.
    """
    if not issues:
        # `valid: false` with no error-level issue is a lint that failed on its own summary line;
        # keep the old wording rather than inventing a reason we do not have.
        return "the resulting problem is invalid."
    shown = [" ".join(str(m).split())[:_REJECTION_ISSUE_CHARS] for m in issues[:_REJECTION_ISSUES_SHOWN]]
    more = len(issues) - len(shown)
    tail = f" (+{more} more in `issues`)" if more > 0 else ""
    joined = " | ".join(shown)
    return f"the resulting problem is invalid: {joined}{tail}"


def _commit_change(
    session,
    op: str,
    op_args: dict[str, Any],
    justification: str,
    references: list[str] | None,
    *,
    model_build: Callable[[], tuple[Any, Any]] | None = None,
    param_edit: Callable[[Any], Any] | None = None,
    # References are optional (only a non-empty justification is required); every caller passes
    # this explicitly, so the default just keeps a future caller from silently making them
    # mandatory again.
    require_references: bool = False,
) -> dict[str, Any]:
    """Apply one model/parameter edit to the working model — WITHOUT fitting or versioning.

    ``model_build()`` returns a brand-new ``(SbmlModel, apply_result)`` that *replaces*
    the model wholesale (both re-authoring an existing model and zero-shotting one when none
    exists — reconciled either way); ``param_edit(problem)`` edits the parameter table in place.
    A ``param_edit`` may accompany a
    model build (``set_model(parameters=[...])``, issue #114): both then land in the SAME
    transaction — the parameter edit runs AFTER the reconcile, so it can target a parameter the
    re-author just introduced, and a rejected entry reverts the re-author too. On any failure the
    model and parameter table are reverted so an edit is atomic.

    Applies and reconciles + validates; the caller scores it (:meth:`session.PetabSession.fit`,
    which also carries the structure-level de-duplication). Returns ``applied: true`` with the
    reconcile/validation/warnings and no fit metrics.
    """
    # Neither of today's two callers can trip this, but with both callbacks None the body applies
    # NOTHING and still appends an `edit` event to the append-only log. A silent no-op in the
    # audit trail is the one failure this layer must not produce.
    if model_build is None and param_edit is None:
        raise ValueError(
            "_commit_change requires at least one callback: a model_build (optionally with a "
            "param_edit), or a param_edit on its own."
        )
    if not isinstance(justification, str) or not justification.strip():
        raise OperationError("A non-empty justification is required for this operation.")
    resolved = resolve_references(session.resolver, references, required=require_references)

    with session.lock:
        old_species = set(R.species_ids(session.problem.model.sbml_model))
        # The edit runs against a CANDIDATE problem, never the live one. Shallow copy (the fixed
        # measurement/condition/observable tables are read-only and shared, as
        # `fitting._build_hierarchical_problem` already does) with its own parameter table, since
        # `reconcile` and `param_edit` rewrite that in place. A rejected edit is then simply a
        # candidate that is never swapped in — there is no unwind, because nothing was wound.
        problem = copy.deepcopy(session.problem)
        # DEEP: nothing is shared with the live problem, so an in-place edit anywhere in the
        # candidate — the parameter table, or the SBML on the parameters-only path, where no fresh
        # model replaces it — cannot reach the working problem or the undo snapshot
        # `working_state` takes. A shallow copy leaves the model object shared, and a mutation
        # through it lands before any check can reject the edit.
        model_sbml_before = None if model_build is not None else problem.model.to_sbml_str()

        apply_result: Any = None
        if model_build is not None:
            candidate, apply_result = model_build()
            problem.model = candidate
            recon = R.reconcile(problem, shipped=session.shipped_parameter_df)
            # An EMPTY model is refused rather than committed — where "empty" is the same
            # predicate `model_authored` reports: no species at all, or no DYNAMICS (neither
            # a reaction nor a rate/assignment rule). Both are unscoreable, and the first is
            # actively dangerous: AMICI's SBML import segfaults on a model with no entities
            # whose observables reference undefined symbols, killing the server process, so
            # nothing downstream gets to reject it gracefully. Refusing here is also what
            # keeps `model_authored` meaningful — it can only ever go False -> True, on the
            # first call that commits real dynamics, never back again.
            #
            # The check runs AFTER the reconcile: a model can arrive non-empty and be pruned
            # to nothing (every species unreferenced by any reaction, rule, IC or fixed
            # table), and re-authoring the content-free seed lands here too. Only an
            # authoring call is checked — a parameters-only edit cannot change the model.
            if not R.has_dynamics(problem.model.sbml_model):
                raise ValidationError(
                    f"Operation {op!r} rejected: the resulting model has no DYNAMICS — "
                    f"{problem.model.sbml_model.getNumSpecies()} species, no reactions and "
                    "no rate/assignment rules, so every state stays at its initial value and "
                    "there is nothing to fit.",
                    issues=[
                        "Author the mechanism itself: at least one reaction (`A -> B; k*A`) "
                        "or a rate/assignment rule that changes a species over time. An "
                        "Antimony string that only DECLARES species and parameters — or "
                        "whose species all prune away as orphans (unused by any reaction, "
                        "rule, initial assignment, or the observable/condition tables) — "
                        "reconciles to a model that cannot move."
                    ],
                )
        else:
            recon = None
        if param_edit is not None:
            # AFTER the reconcile: the parameter table only holds the re-authored model's
            # parameters once reconcile has run, so a bundled bound edit can target a symbol
            # this very (re-)author introduced — and an entry naming a parameter the re-author
            # dropped is rejected loudly instead of silently doing nothing.
            param_result = param_edit(problem)
            apply_result = (
                {"model": apply_result, "parameters": param_result}
                if recon is not None else param_result
            )
        # A parameters-only call must not change the model: it never runs the reconcile or the
        # has-dynamics refusal, so a model edit smuggled through it would reach a committed
        # version unchecked. The live problem is safe either way (the candidate is a deep copy);
        # this is the operation's contract, not a containment measure.
        if model_sbml_before is not None and problem.model.to_sbml_str() != model_sbml_before:
            raise ValidationError(
                f"Operation {op!r} rejected: a parameters-only edit changed the MODEL, which "
                "bypasses the SBML<->PEtab reconcile every authoring call runs. Author model "
                "changes through `antimony`."
            )
        # A parameter that ships FIXED and is a mechanistic/structural constant (a
        # conservation total, a known rate) must not be removed or freed by an edit —
        # validation rejects either. Best-effort: an unreadable v0 yields an empty set.
        try:
            protected_fixed = session.protected_fixed_parameter_ids()
        except Exception:  # noqa: BLE001 - never let the lookup break an edit
            protected_fixed = None
        report = validate_problem(problem, protected_fixed_ids=protected_fixed)
        if not report["valid"]:
            hard = [i["message"] for i in report["issues"]
                    if i["level"] in ("error", "critical")]
            raise ValidationError(
                f"Operation {op!r} rejected: {_rejection_summary(hard)}",
                issues=hard,
            )

        warnings = (_parsimony_warnings(problem, old_species) if recon is not None else [])
        # Warnings raised by the edit callback itself (e.g. change_parameters' advisory sigma cap)
        # must reach the CALLER, not just the provenance log — an advisory nobody reads is the same
        # as no guard at all. `recon is None` for every parameter edit, so the parsimony list above
        # is always empty there and these would otherwise be dropped.
        # A bundled `set_model(parameters=[...])` (#117) nests the parameter result under
        # "parameters", so reading only the top level silently dropped the advisory on exactly the
        # call that re-authors the model AND widens the noise box together.
        for _source in (apply_result, (apply_result or {}).get("parameters")
                        if isinstance(apply_result, dict) else None):
            if isinstance(_source, dict):
                warnings = warnings + [w for w in (_source.get("warnings") or []) if w]
        # The edit's half of the message stack (see :mod:`messages`). The lint's WARNING-level
        # findings join it here: they used to live only inside the `validation` report, which the
        # result no longer carries, so they were reaching nobody at all.
        edit_messages = M.warnings_from(M.SOURCE_EDIT, warnings) + M.warnings_from(
            M.SOURCE_VALIDATION,
            [i["message"] for i in report["issues"] if i.get("level") == "warning"],
        )
        param_diff = {
            "added": recon["parameters"]["added"] if recon else [],
            "removed": recon["parameters"]["removed"] if recon else [],
            "pruned_species": recon["pruned"]["species"] if recon else [],
        }
        # Per-species initial-condition account: for every species, whether its declared t=0
        # is FIXED (survives into the simulated t=0 state), ESTIMATED (the fit sets it), or
        # overridden per-condition — so the caller can see immediately whether a t=0 fix took,
        # instead of guessing IC naming schemes that all fit identically (issue #57). Cheap and
        # analytic; `newly_added` marks the ICs this op introduced. Unconditional — there is
        # always a model (see `session.has_model`).
        initial_conditions = R.initial_conditions_report(
            problem, added=set(param_diff["added"]))
        # Log the edit. No version — versions are fit snapshots (see `session.fit`). Written
        # before the fit so a change that fails to score still leaves a trace, and because
        # `references` (resolved PubMed metadata) lives nowhere else.
        event = session.provenance.append_event({
            "type": "edit",
            "operation": op,
            "args": op_args,
            "result": apply_result if isinstance(apply_result, (str, dict, list)) else None,
            "justification": justification,
            "references": resolved,
            "parameter_diff": param_diff,
            "warnings": warnings,
        })

        # LAST, once nothing left can raise: the candidate becomes the working problem. Everything
        # above reads the local `problem`, so the swap has no reason to happen earlier — and every
        # statement above it (the parsimony walk, the initial-condition report, the event write)
        # can throw. A throw after the swap would leave the live problem changed while the caller
        # reports `applied: false`, and `make_changes` returns on that without reverting.
        session.problem = problem

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
        "messages": edit_messages,
        "parameter_diff": param_diff,
        "event_timestamp": event["timestamp"],
        # What this edit was, for `session.fit` to record on the fit event. Passed forward rather
        # than parked on the session: the change and its score are one call, so there is no
        # unscored-edit state for the fit to read back.
        "staged_edit": {"operation": op, "args": op_args, "justification": justification},
    }


# -- public operations: model ----------------------------------------------
def set_model(
    session, antimony: str, justification: str,
    references: list[str] | None = None,
    parameters: list[dict[str, Any]] | None = None,
    restore_parameter_df: Any | None = None,
) -> dict[str, Any]:
    """Author the WHOLE model from Antimony — the structural edit path of ``make_changes``.

    This BUILDS the first model when the problem ships model-less (``model_authored:
    false``) AND re-authors/replaces an existing model wholesale. Every structural change — adding
    or removing a reaction, changing a rate law — is made by re-authoring the whole model in one
    coherent step. Like every edit it reconciles SBML↔PEtab (parametric ICs + parameter-table
    sync) and lints, reverting on failure so a bad (re-)author changes nothing; the caller scores
    it, and the de-duplication that keeps an identical structure from being fit twice lives there.

    Because the whole model is replaced, reconcile drops the parameter-table rows of any
    parameter the new model no longer declares. That silently discards a previously-fitted
    value, so this warns when the re-author removes a previously-ESTIMATED parameter (fixed
    noise/initial-condition parameters and anything still referenced by the observable/noise
    or condition tables are preserved). Dropping (or reintroducing as estimated) a shipped-FIXED
    mechanistic constant — a conservation total, a known rate — is a harder case: the edit
    pipeline's validation REJECTS it outright (see ``validation._fixed_constant_role_issues``).

    ``parameters`` optionally carries the SAME per-parameter entries :func:`change_parameters`
    takes (issue #114). They are applied in the same transaction as the re-author, AFTER the reconcile —
    so an entry may target a parameter this author introduces, and a rejected entry reverts the
    re-author too. ``None``/``[]`` means "no parameter edit"; the whole bundle is recorded as ONE
    edit event, since it is one logical change.
    """
    if not isinstance(antimony, str) or not antimony.strip():
        raise OperationError("make_changes requires a non-empty Antimony model string when `antimony` is given.")

    normalized = (
        _normalize_parameter_changes(parameters, arg="parameters")
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
                    "make_changes rejected: a reaction reuses a parameter/symbol as a stoichiometry "
                    "coefficient (e.g. on both sides), which Antimony cannot represent.",
                    issues=[i["message"] for i in conflict],
                ) from exc
            raise ModelUpdateError(
                f"Failed to parse antimony model: {exc}.{_antimony_parse_hint(antimony, exc)}"
            ) from exc
        candidate.rel_path = getattr(session.problem.model, "rel_path", None) or "model.xml"
        sm = candidate.sbml_model
        # A parameter written as a reaction stoichiometry coefficient (e.g. `-> N_burst X`) is
        # frozen to a constant and dropped by the Antimony→SBML conversion; libantimony leaves the
        # dropped symbol as the SpeciesReference id, so catch it on the freshly-converted candidate
        # before it is committed (see validation.frozen_parametric_stoichiometry_issues, issue #89).
        stoich_issues = frozen_parametric_stoichiometry_issues(sm)
        if stoich_issues:
            raise ValidationError(
                "make_changes rejected: a reaction uses a parameter/symbol as a stoichiometry "
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
        op_args["parameters"] = _parameters_arg(normalized)
    result = _commit_change(
        session, "make_changes", op_args, justification, references,
        model_build=build,
        param_edit=_composed_parameter_edit(session, normalized, restore_parameter_df),
        require_references=False,
    )

    # Guardrail: a re-author that no longer declares a previously-estimated parameter drops
    # its row (and its fitted value). Surface that — it is the one real footgun of a wholesale
    # replace; fixed/table-referenced parameters are not affected.
    removed = ((result.get("reconcile") or {}).get("parameters") or {}).get("removed") or []
    dropped = sorted(p for p in removed if str(p) in prior_estimated)
    if dropped:
        note = (f"the re-author dropped previously-estimated parameter(s) {dropped}: their fitted "
                "values are gone. Re-declare them in the Antimony if that was unintended.")
        result["warnings"] = list(result.get("warnings") or []) + [note]
        result["messages"] = list(result.get("messages") or []) + [
            M.warning(M.SOURCE_EDIT, note)]
    return result


# -- public operations: parameters (justification required, references optional) --
def _antimony_parse_hint(text: str, exc: Exception) -> str:
    """Echo the offending SOURCE LINE, plus a likely cause when the line's shape is recognisable.

    Antimony reports a line NUMBER and the unexpected token, never the line — so the agent is left
    matching a bison tail against a model it has to re-read. 267 parse rejections in the August
    corpus, 11 traces, 4 model families, 7 distinct tails.

    Gated on the source LINE rather than the token, because token→construct is not 1:1:
    `parameter V_init;` reports "unexpected ';'", which the token alone misdiagnoses. Hints are
    worded "likely cause" and never as a verdict; the echo alone is useful even when none matches.
    """
    m = re.search(r"line (\d+)", str(exc))
    if not m:
        return ""
    lines = text.split("\n")
    i = int(m.group(1))
    if not (1 <= i <= len(lines)):
        return ""
    src = lines[i - 1].strip()
    parts = [f" line {i} reads: {src}"]
    if re.match(r"^(param|parameter)\b", src):
        parts.append("likely cause: `parameter`/`param` is not an Antimony keyword — a parameter "
                     "is a plain assignment (`NN = 480;`); a comma list needs a real keyword "
                     "(`const a = 1, b = 2;`). Keep `species`, `compartment`, `var`, `const`.")
    elif re.search(r"^\w+\s*:.*;.*\w+\s*=", src):
        parts.append("likely cause: a rate law is a bare expression and cannot be named — write "
                     "`R: -> V; k_prod;`, not `R: -> V; v_prod = k_prod;`.")
    elif re.search(r"(->|=>)", src) and re.search(r"\b\w+\s*\*\s*\w+", src.split(";")[0]):
        parts.append("likely cause: stoichiometry is a numeric literal separated by a space, not "
                     "a `*` — and never a parameter. A burst size belongs in the rate law: "
                     "`r: -> V; NN*delta*I;`.")
    elif re.match(r"^\w+\s*;", src):
        parts.append("likely cause: a name on its own is not a declaration — use `species V;`, "
                     "`var gamma;`, or an assignment `gamma = 0.1;`.")
    return " ".join(parts)


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
                # The bare rejection is the ONE mode in the August trace corpus where the agent
                # could not recover from the message it was given: it names `nominalValue`, which
                # no argument can set, so a run alternated this rejection with an unknown-key
                # rejection 41 times each and spent its whole budget without a single fit. Name
                # the exits that actually work.
                if field == "lowerBound":
                    hint = (" Bounds are LINEAR values, not log10 coordinates: a rate spanning "
                            "1e-4..1e2 is lower_bound=1e-4, upper_bound=1e2 — never -4/2.")
                else:
                    hint = (
                        " `nominalValue` is not settable here (allowed keys: parameter_id, "
                        "lower_bound, upper_bound, scale) — a parameter's VALUE comes from the "
                        "model text or the fit. An initial-condition row is created from the "
                        "initial value the model declares AT THE MOMENT IT IS FIRST CREATED: 0 "
                        "(or no initial value) means FIXED at 0, 1 means FIXED at 1, any other "
                        "positive value means ESTIMATED on log10. A row already created at 0 does "
                        "NOT re-open when you later declare a positive initial — only a species "
                        "authored under a DIFFERENT name gets a fresh row. So either drop this "
                        "scale entry (a species that starts empty is correctly fixed at 0), or "
                        "author the species under a different name with a positive initial value."
                    )
                raise OperationError(
                    f"Cannot use {scale} scale for {parameter_id!r}: {field}={val} must be > 0."
                    + hint
                )


def _warn_sigma_bound_widening(problem, parameter_id: str, upper_bound: float) -> str | None:
    """Advisory warning when an ESTIMATED noise sigma's upperBound is widened past its data-driven
    cap (see ``R.sigma_upper_bound_cap``), or ``None``. A FIXED sigma is ignored: its bounds don't
    affect a fit it is never optimized over.

    WARNS rather than rejects (#118). The cap is a fraction of the observable's MARGINAL data std,
    which conflates variance the model explains with variance it can't: across the benchmark, the
    true models' own sigmas span ~0.0004-0.59 of that std, so any band wide enough to be safe is
    too wide to be decisive. On a real problem there is no reference to check it against, so a hard
    rejection could block a legitimate edit with no recourse. The load-bearing evidence for "sigma
    is absorbing misfit" is the RESIDUAL STRUCTURE (the fit result's residual_structure verdict: the
    sign/runs tests plus the replicate lack-of-fit test), which is
    transform-invariant and needs no ground truth — this warning just points at it."""
    if parameter_id not in R.plausibility_sigma_ids(problem):
        return None
    if not R.is_parameter_estimated(problem, parameter_id):
        return None
    # No fit is in hand at bound-edit time, so a per-gel amplitude resolves from the table's
    # NOMINAL gains while the commit-time report resolves it from the fit (see
    # reconcile.sigma_data_std). The two caps therefore differ by fitted-gain/nominal-gain on a
    # problem with gains, and this one goes None (advisory silently off) if such a gain has no
    # nominal at all. Left as-is deliberately: this is an ADVISORY warning, not the gate, so it
    # fails open by design, and the alternative — threading a stale previous fit in here — would
    # make the advice depend on which fit last ran.
    cap = R.sigma_upper_bound_cap(problem, parameter_id)
    if cap is None or float(upper_bound) <= cap:
        return None
    # Only warn about an actual WIDENING. Without comparing to the CURRENT bound this fires on a
    # TIGHTENING that merely lands above the cap — e.g. perelson ships [1e-10, 1e10] against a cap
    # of ~0.31, so an agent correctly narrowing to [1e-3, 1.0] would be told its improvement is a
    # dangerous widening. A bound that was already past the cap and is moving DOWN is progress.
    try:
        current = float(problem.parameter_df.loc[parameter_id, UPPER_BOUND])
    except (KeyError, TypeError, ValueError):
        current = None
    if current is not None and current == current and float(upper_bound) <= current:
        return None
    return (
        f"WARNING: widening the upperBound of {parameter_id!r} to {upper_bound:.6g} takes it past "
        f"{cap:.6g} ({NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD:.0%} of this observable's measurement "
        "std, on the scale its noise is additive on). A sigma free to grow can make an "
        "under-mechanistic model look statistically adequate. This is ADVISORY — check the fit "
        "result's residual_structure verdict: if it flags an observable (sign/runs tests, "
        "or a replicated design point the model cannot reach), the mechanism is missing something "
        "and widening the noise will only hide it (get_fit_diagnosis has the detail behind it)."
    )


def _pinned_side(session, problem, parameter_id: str) -> str | None:
    """``"lower"``/``"upper"`` if the parameter's last fitted value sits AT that bound (it
    re-pinned), else ``None``. Reads the current HEAD fit from ``session.simulation`` against
    the parameter table's current bounds — the state as it is BEFORE this bounds edit applies.
    ``None`` (no block) whenever there is no fit to judge against."""
    from . import fitting

    fitted = (session.simulation or {}).get("fitted_parameters")
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
    prov = session.provenance
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
    prev_nllh: float | None = None
    # Anchor on the table the problem SHIPPED, so the walk sees a transition into the first
    # version instead of consuming it to establish a baseline.
    try:
        prev_bound = float(session.shipped_parameter_df.loc[parameter_id, bound_col])
    except Exception:  # noqa: BLE001 - parameter not in the shipped table
        prev_bound = None
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
    ``get_fit_diagnosis(sections="parameters")``) is UNIDENTIFIABLE, not under-explored: each widening just lets the
    optimum slide to the new edge and re-pin, chasing the search space toward infinity for ~0 BIC
    This is the hard backstop behind the widen-at-most-once guidance on the tool itself. It applies to NOISE SIGMAS TOO: they used to be exempt because
    ``_reject_sigma_bound_widening`` rejected them on a stricter data-driven cap, but that cap is
    now only advisory (:func:`_warn_sigma_bound_widening`, #118 — it compares against the marginal
    data std, which no fixed band can hold), so exempting them here would leave the one parameter
    class that can buy adequacy by inflating with NO hard limit at all. This only ever REJECTS — it
    never rewrites a bound — and a missing fit / provenance simply means no block."""
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
        "get_fit_diagnosis(sections=\"parameters\")) is UNIDENTIFIABLE, not under-explored — widening chases it toward "
        "infinity while it keeps re-pinning. Instead re-author with make_changes to REMOVE its "
        "reaction or use its LIMITING form (make the reaction irreversible / the step "
        "instantaneous); if neither is possible, LEAVE it pinned — do not widen again or "
        "hard-code a value (there is no value/estimate tool, and a literal is just a magic "
        "number — values come only from the fit)."
    )


def _normalize_parameter_changes(
    changes: list[dict[str, Any]], label: str = "make_changes", arg: str = "parameters",
) -> list[dict[str, Any]]:
    """Validate + normalize a ``parameters`` list into ``{parameter_id, lower_bound, upper_bound,
    scale}`` entries (missing fields as ``None``), raising :class:`OperationError` on anything
    malformed. Shared by both edit paths so they enforce exactly the same entry contract (issue
    #114); ``label``/``arg`` name the offending argument in the rejection message, which is why
    they default to the TOOL's names rather than these functions'."""
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


def _restore_parameter_edit(stored_df, skip: set[str] | None = None) -> Callable[[Any], dict[str, Any]]:
    """A ``param_edit`` that puts a STORED version's search space back.

    Used by a `base_version=N` fork so that going back to a version restores its bounds, scales
    and nominal values along with its model — the prompt used to tell the agent that
    `base_version` "rolls back the MODEL only" and to re-send `parameters` by hand, which meant a
    fork silently inherited whatever the abandoned branch had widened, and re-started the search
    from the abandoned branch's optimum.

    Deliberately TOLERANT of a changed symbol set: it restores only ids the post-reconcile table
    still has. A fork that removes a reaction drops that reaction's parameters, and a strict
    `parameters` entry for a now-absent id would reject the whole call — turning a working fork
    into a failed one. Ids the stored table lacks (introduced by this call's operations) keep
    whatever the reconcile minted.

    The widening backstops are deliberately NOT applied: restoring a state that already existed and
    was already fit is not a widening request, and `_reject_repeated_widening` would refuse it.
    """
    from .parameters import update_parameters
    from petab.v1.C import LOWER_BOUND, NOMINAL_VALUE, PARAMETER_SCALE, UPPER_BOUND

    # BOUNDS, SCALE, AND NOMINAL VALUE — the search space and the point the fit starts from.
    #
    # nominalValue is included DELIBERATELY, and it is the half that does the most work. The fit
    # writes each estimated parameter's fitted value back into nominalValue, and half the
    # multistart is seeded from those values (see fitting.warm_start_guesses), so restoring them is
    # what makes a fork actually resume from the version it names: without it the fork keeps the
    # ABANDONED branch's optimum as its warm-start point, which is the opposite of rolling back.
    # Two consequences worth knowing rather than discovering:
    #   * the dedup fingerprint folds in nominalValue, so a fork that restores it is compared
    #     against prior versions on the restored point, not the abandoned one;
    #   * if the restored point is non-finite under the fork's NEW structure, `objective_before`
    #     gates warm starts off for that fit and it runs cold — a slower fit, not a wrong one.
    #
    # NOT estimate — and the reason is NOT "the caller cannot override it", since that is equally
    # true of nominalValue. It is what surviving the fit means for each. nominalValue is a STARTING
    # POINT: the fit immediately writes over it, so restoring it steers one search and then stops
    # mattering. `estimate` is a STRUCTURAL choice that persists: restoring a 0 would re-fix a
    # parameter the agent had freed, with no argument able to free it again and nothing in the
    # payload naming what happened. One is a hint, the other is a lock.
    # `skip` (the ids this call's own `parameters` claim) withholds only the SEARCH-SPACE columns.
    # nominalValue is restored for every id regardless, because the caller has no way to ask for a
    # starting point: `ParameterChange` forbids the key outright (see `_require_positive_for_log`,
    # which exists to explain that rejection). Letting `skip` cover it would mean that naming one
    # bound in a fork silently opted that parameter out of the rollback's warm start — a
    # search-space preference quietly costing you the starting point you asked for by forking.
    SPACE = (LOWER_BOUND, UPPER_BOUND, PARAMETER_SCALE)
    POINT = (NOMINAL_VALUE,)

    def edit(problem):
        live = problem.parameter_df
        updates, restored, skipped, point_only = [], [], [], []
        for pid in stored_df.index:
            pid = str(pid)
            if pid not in live.index:
                skipped.append(pid)
                continue
            # theirs wins on the space they named; the point is restored either way
            claimed = bool(skip) and pid in skip
            cols = POINT if claimed else SPACE + POINT
            upd = {"id": pid}
            for c in cols:
                if c not in stored_df.columns:
                    continue
                raw = stored_df.loc[pid, c]
                # CAST out of pandas. `get_parameter_df` is a read_csv, so `estimate` comes back
                # as numpy.int64 — and `isinstance(np.int64(1), (int, float))` is False on numpy 2,
                # so `_coerce_estimate_int` would reject it and roll the whole fork back with a
                # message blaming the caller's parameters. Skip NaN (an absent optional column)
                # rather than writing it.
                if raw is None or raw != raw:
                    continue
                upd[c] = str(raw) if c == PARAMETER_SCALE else float(raw)
            if len(upd) > 1:
                updates.append(upd)
                (point_only if claimed else restored).append(pid)
        if updates:
            update_parameters(problem, updates)
        out = {"restored_parameters": restored}
        if point_only:
            # Named in this call's `parameters`, so the bounds are the caller's; only the
            # starting point came back. Reported apart so the distinction is legible.
            out["restored_start_only"] = point_only
        if skipped:
            out["parameters_not_in_new_model"] = skipped
        return out

    return edit


def _composed_parameter_edit(session, normalized, stored_df):
    """Restore a stored search space FIRST, then apply this call's explicit `parameters` on top.

    Order is the contract: an explicit entry in the same call must win over the restored value, so
    a caller can fork a version AND change one bound in one transaction.
    """
    have_stored = stored_df is not None and len(stored_df)
    apply_new = _parameter_edit(session, normalized) if normalized else None
    if not have_stored:
        return apply_new
    # The caller's OWN entries must be measured against the PRISTINE table: `_parameter_edit`'s
    # widening backstops read the pre-edit bound (`_reject_repeated_widening` off the live table,
    # `_pinned_side` against HEAD's fitted values), and restoring first would hand them the old
    # version's bounds as the baseline — so a third widening could slip past the guard, or an
    # ordinary entry be flagged as one. So: caller first, then restore only the parameters the
    # caller did not mention. "An explicit entry wins" still holds, by exclusion rather than by
    # overwrite.
    claimed = {c["parameter_id"] for c in (normalized or [])}
    restore = _restore_parameter_edit(stored_df, skip=claimed)
    if apply_new is None:
        return restore

    def edit(problem):
        first = apply_new(problem) or {}
        second = restore(problem) or {}
        merged = dict(second)
        merged.update(first)
        return merged

    return edit


def _parameter_edit(session, normalized: list[dict[str, Any]]) -> Callable[[Any], dict[str, Any]]:
    """Build the ``param_edit(problem)`` callback that applies ``normalized`` (from
    :func:`_normalize_parameter_changes`) as ONE transactional write, with every per-parameter
    backstop. Shared by ``change_parameters`` and ``set_model(parameters=[...])``."""
    from .parameters import update_parameters

    def edit(problem):
        updates: list[dict[str, Any]] = []
        scale_checks: list[tuple[str, str]] = []  # (pid, scale) to verify AFTER the batched update
        changed = []
        warnings: list[str] = []
        for ch in normalized:
            pid, lb, ub, scale = ch["parameter_id"], ch["lower_bound"], ch["upper_bound"], ch["scale"]
            update: dict[str, Any] = {"id": pid}
            if ub is not None:
                # Widening backstops read the pre-edit bound, so they must run BEFORE any update.
                # Entries are distinct parameters, so checking them all against the pristine table
                # (before the single batched write below) is equivalent to interleaving.
                # The sigma cap only WARNS (#118) — it is a marginal-std heuristic with no ground
                # truth to check it against on a real problem; the repeated-widening backstop below
                # is unidentifiability evidence and still rejects.
                sigma_warning = _warn_sigma_bound_widening(problem, pid, ub)
                if sigma_warning:
                    warnings.append(sigma_warning)
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
        out: dict[str, Any] = {"changed": changed}
        if warnings:
            out["warnings"] = warnings   # advisory only; the edit was applied
        return out

    return edit


def _parameters_arg(normalized: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The normalized entries as they go into the provenance event args (``None``s dropped).

    Recorded under the same key both editors accept (``parameters``), so a provenance event reads
    back as the call that produced it."""
    return [{k: v for k, v in c.items() if v is not None} for c in normalized]


def change_parameters(
    session,
    parameters: list[dict[str, Any]],
    justification: str,
    references: list[str] | None = None,
) -> dict[str, Any]:
    """Batch-edit the bounds and/or scale of one or more parameters in ONE atomic edit.

    ``parameters`` is a non-empty list of per-parameter dicts, each naming ``parameter_id`` and any
    of ``lower_bound`` / ``upper_bound`` (a bound edit — give BOTH or neither) and ``scale``
    ('lin'|'log'|'log10'). Group everything that belongs to one coherent retuning into a single
    call. Every per-parameter backstop still applies (a re-pinned parameter's 3rd widening, a
    noise-sigma cap, a log scale on a non-positive box); because the whole batch is atomic, one
    rejected entry reverts ALL of them and nothing changes. An EDIT: mutates the parameter table
    but does NOT fit — the fit that scores it is wired at the server layer, not here."""
    normalized = _normalize_parameter_changes(parameters)
    return _commit_change(
        session, "make_changes", {"parameters": _parameters_arg(normalized)},
        justification, references,
        param_edit=_parameter_edit(session, normalized), require_references=False,
    )


# -- edit + fit, as one all-or-nothing change ------------------------------
def working_state(session) -> dict[str, Any]:
    """Snapshot the working state, so a FIT that produces no version can be undone.

    A rejected edit needs no unwind — it builds a candidate problem and never swaps it in (see
    :func:`_commit_change`). This snapshot covers the step after it, the fit: the candidate IS the
    working problem by then, and a fit that produces no version has to put the old one back.

    Holding the problem OBJECT is enough because the edit replaces the reference rather than
    mutating it. The last fit's ``simulation`` / ``optimization`` payloads come too — what
    ``get_fit_summary`` and every residual/diagnostic read report. ``fit_and_simulate`` REBINDS
    them rather than mutating, so keeping the references restores the score with the model;
    otherwise a fit that ran and then failed to snapshot would leave the reverted-away
    candidate's metrics attributed to the restored model.
    """
    with session.lock:
        return {
            "problem": session.problem,
            "simulation": session.simulation,
            "optimization": session.optimization,
        }


def revert_working_state(session, snapshot: dict[str, Any], reason: str) -> None:
    """Undo an applied edit whose fit did not produce a version, and log the revert.

    The invariant: a change is either scored or gone, so the working model is never a state
    nothing verified. The `edit` event is not erased — the log is append-only and the edit did
    happen — a ``revert`` event is appended after it.

    The last fit's ``simulation``/``optimization`` go back too: a fit that simulated before
    failing has already overwritten both, and restoring only the model would leave
    ``get_fit_summary``, the residual diagnostics and the SQL ``simulations`` table describing a
    candidate that is no longer the working model.
    """
    with session.lock:
        session.problem = snapshot["problem"]
        session.simulation = snapshot["simulation"]
        session.optimization = snapshot["optimization"]
        session.provenance.append_event({
            "type": "revert",
            "operation": "make_changes",
            "reason": reason,
        })



# Keys of the edit half that survive into a `make_changes` result: its DIAGNOSTICS, merged with
# the fit's below. Everything else the edit produces is dropped — it rides on EVERY call into a
# history nothing prunes, and none of it is a score. Readable on demand instead:
# `get_model_structure` carries the reconciled model and the initial-condition account,
# `get_parameters` each row's role and status.
_EDIT_KEYS_KEPT = ("messages",)


def bundle_result(
    session, edit_result: dict[str, Any], fit_result: dict[str, Any],
    head_before: int,
) -> dict[str, Any]:
    """The result of one ``make_changes`` call: a flag, a version, and the fit — flat.

    The fit is FLATTENED to the top level rather than nested under ``fit``, because after the
    collapse there is no second thing in the result to disambiguate it from: a call is a change
    and its score, so the score IS the result. That also makes a score read off ``make_changes``
    name its fields exactly as one read off ``get_fit_summary`` or ``restore_version``.

    ``applied`` is the state flag: whether the change is IN the working model when the call
    returns. It is not "did the edit apply" — an edit that applies and then fails to fit is
    reverted (see :func:`revert_working_state`), so that call reports ``applied: false``, and the
    agent never has to reason about a half-done state. ``committed`` is the separate question of
    whether a VERSION was created; the two differ only for the de-dup no-op, where the change is
    in place but is byte-identical to an existing version.

    Both halves' diagnostics are merged into ONE ordered ``messages`` stack (see :mod:`messages`) —
    the edit's advisories and lint warnings first, then the fit's — so a caller reads every
    warning and the error that stopped the call from a single place, each tagged with the step it
    came from.

    On a failure the message does NOT prescribe a fix, it forwards ``error`` — because that error
    is not always "the model is broken". The identical-structure loop guard reaches the same
    branch, and its message says the opposite (stop re-sending this structure, make a genuinely
    different change); a top-level "repair it and try again" would send the agent straight
    back into the loop the guard exists to break, and the agent reads the message first.

    Whether a version was created is decided by comparing HEAD against ``head_before`` — the only
    signal that survives a failure between the snapshot and the end of ``session.fit``.
    """
    committed = bool(fit_result.get("committed"))
    no_change = bool(fit_result.get("no_change"))
    version = fit_result.get("version")
    head_now = session.provenance.current_version
    snapshotted = head_now != head_before
    if committed:
        applied = True
        message = f"Change applied and scored as version {version}."
    elif no_change:
        applied = True
        message = (
            "Change applied, but the fit was a no-op: the resulting model is identical to version "
            f"{fit_result.get('duplicate_of_version')}, so its stored metrics stand. Make a "
            "genuinely different change; that version is kept and scored regardless."
        )
    elif snapshotted:
        # The fit broke only AFTER snapshotting, so HEAD moved: report where it landed rather than
        # asking for a re-fit that would be a de-dup no-op against the version just created.
        applied = True
        version = version if version is not None else head_now
        message = (
            "Change applied and a version was snapshotted, but the fit did not finish cleanly, so "
            f"this call carries no metrics: the working model is now version {version}. Read "
            "`error`, then read that version's score with list_versions before making another "
            "change — do NOT re-send it, the model is already scored."
        )
    else:
        applied = False
        version = head_now
        message = (
            "Change REVERTED: it could not be scored, so it was undone — the working model is "
            f"unchanged, still version {head_now}, and nothing is left half-done. The fit "
            f"reported: {fit_result.get('error') or 'no error message'} — ACT ON THAT rather than "
            "assuming the model is broken: it says whether to fix the change and send it again, "
            "or to stop re-sending this structure and make a genuinely different change instead."
        )
    # Pipeline order: the edit's messages, then the fit's. `run_fit` / `edit_then_fit` already
    # shape a rejected fit as error entries; a fit that RAN reports its advisories as the flat
    # `warnings` the de-dup produces, so both shapes are folded in here.
    messages = (
        list(edit_result.get("messages") or [])
        + list(fit_result.get("messages") or [])
        + M.warnings_from(M.SOURCE_FIT, fit_result.get("warnings"))
    )
    result = {
        **{k: edit_result[k] for k in _EDIT_KEYS_KEPT if k in edit_result},
        **{k: v for k, v in fit_result.items()
           if k not in ("committed", "version", "warnings", "messages")},
        "applied": applied,
        "committed": committed,
        "version": version,
        "messages": messages,
        "message": message,
    }
    return result
