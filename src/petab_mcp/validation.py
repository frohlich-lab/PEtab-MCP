"""Problem validation: PEtab v2 linting + libsbml SBML consistency.

libpetab's ``Problem.validate()`` returns structured issues (level, message, the check that
raised it), run here on :meth:`problem.WorkingProblem.to_petab`, and the same lint enforces the
invariant that the model must satisfy the fixed tables (an observable/condition/parameter
referencing a missing entity is an error).

petab's lint only checks the PEtab *tables* against the model, not the SBML document
itself, so we ALSO run libsbml's ``checkConsistency`` on the candidate model and fold its
findings into the same ``issues`` list. Structural SBML problems (general/identifier/MathML/
overdetermined consistency) become hard errors that gate the commit — surfaced to the agent
pre-compile with a clear message instead of an opaque AMICI codegen crash. Unit-consistency,
modeling-practice, and SBO checks are turned OFF: these are unit-less mechanistic models, so
those categories would flag essentially every reaction/parameter and drown the real signal.
"""

from __future__ import annotations

from typing import Any

from petab.v2.C import CONDITION_ID, ESTIMATE, TARGET_ID, TARGET_VALUE
from petab.v2.lint import ValidationIssueSeverity

from .problem import as_bool
from .reconcile import (
    _ast_names,
    condition_target_symbols,
    noise_sigma_ids,
    _model_math_names,
    _reaction_species,
    has_dynamics,
    observation_model_symbols,
    parameter_ids,
    species_ids,
)

# libsbml consistency categories that are pure noise for unit-less mechanistic models — we
# neither gate nor report them (every species/parameter would otherwise flag a unit warning).
_SBML_SILENCED_CATEGORIES = (
    "LIBSBML_CAT_UNITS_CONSISTENCY",
    "LIBSBML_CAT_MODELING_PRACTICE",
    "LIBSBML_CAT_SBO_CONSISTENCY",
)
_SBML_MSG_CAP = 300  # keep a single SBML message from bloating the (re-sent) tool result


# petab lint checks that say nothing about THIS EDIT: they warn about the FIXED design (a
# condition or experiment no measurement uses — fujita ships its held-out pulses in the condition
# table), which no call can change. Promoting them would put a fixed, unactionable string on the
# `warnings` of EVERY make_changes result, into a history nothing prunes, so they are dropped at
# the capture site and reach neither `issues` nor the warning COUNT. Warnings only: an ERROR from
# any check is kept.
_DESIGN_ONLY_CHECKS = frozenset({"CheckUnusedConditions", "CheckUnusedExperiments"})


def _sbml_consistency_issues(problem) -> list[dict[str, Any]]:
    """libsbml ``checkConsistency`` on the candidate SBML, as ``issues``.

    ERROR/FATAL findings are hard errors (they gate the commit); WARNING findings are
    advisory. The noisy categories in ``_SBML_SILENCED_CATEGORIES`` are disabled first.
    Best-effort: any failure here returns ``[]`` rather than breaking validation.
    """
    try:
        import libsbml

        sbml_model = getattr(problem.model, "sbml_model", None)
        doc = sbml_model.getSBMLDocument() if sbml_model is not None else None
        if doc is None:
            return []
        for cat in _SBML_SILENCED_CATEGORIES:
            code = getattr(libsbml, cat, None)
            if code is not None:
                doc.setConsistencyChecks(code, False)
        doc.getErrorLog().clearLog()  # so we read only THIS consistency pass
        doc.checkConsistency()

        issues: list[dict[str, Any]] = []
        for i in range(doc.getNumErrors()):
            err = doc.getError(i)
            severity = err.getSeverity()
            if severity >= libsbml.LIBSBML_SEV_ERROR:
                level = "error"
            elif severity == libsbml.LIBSBML_SEV_WARNING:
                level = "warning"
            else:
                continue  # skip INFO
            message = (err.getShortMessage() or err.getMessage() or "").strip()
            if len(message) > _SBML_MSG_CAP:
                message = message[:_SBML_MSG_CAP] + "…"
            issues.append(
                {
                    "level": level,
                    "message": message,
                    "check": f"sbml:{err.getCategoryAsString()}",
                }
            )
        return issues
    except Exception:  # noqa: BLE001 - validation must never crash on the SBML lint
        return []


def _observation_model_reuse_issues(problem) -> list[dict[str, Any]]:
    """Flag a model that reuses a fixed observation-model parameter (an observable-scaling factor
    or noise sigma, resolved through the observable/noise formulas or their measurement-level
    ``observableParameters``/``noiseParameters`` overrides) as a mechanistic quantity inside a
    reaction, rule, or initial assignment. Three-way outcome, see below: a hard error for a NOISE
    SIGMA or for a symbol acting AS a reaction's rate constant, a warning for legitimate reuse.

    A benchmark sweep caught exactly this: two independent runs wired the Brannmark scenario's
    four fixed observable-scaling factors directly into their reaction rate laws instead of
    introducing the ~14 real kinetic parameters the scenario expects, ending up with a
    severely under-mechanistic model that still passed its own chi2 goodness-of-fit check
    (with help from noise sigmas inflated 20-60x). Gate this at commit time rather than relying
    on an agent to notice: reusing one of these symbols does not "cost" anything on its own
    metrics, so nothing else in the pipeline would catch it.

    Blanket rejection is wrong, because a published problem legitimately reuses these symbols: the
    SAME physical quantity is often both a model quantity and the observable's normaliser.
    Bachmann_MSB2011 is the reference case — the receptor/SHP1/STAT5 total abundances
    (``init_EpoRJAK2``, ``init_SHP1``, ``init_STAT5``) set the initial conditions AND divide the
    relative pEpoR/pJAK2/tSTAT5 observables, and the CIS/SOCS3 equilibrium constants (``CISEqc``,
    ``SOCS3Eqc``, ``CISRNAEqc``, ``SOCS3RNAEqc``) scale both the species and their readouts. Making
    that an error rejects the published mechanism itself, and with it any agent that finds it.

    Restricting the exemption to INITIAL ASSIGNMENTS does not work either: all seven of bachmann's
    symbols also appear in rate laws, and two of them (``CISRNAEqc``, ``SOCS3RNAEqc``) have no
    initial assignment at all. Nor does "denominator only" — ``CISEqc`` and ``CISRNAEqc`` appear as
    numerator factors. What actually separates the two cases is the symbol's ROLE:

    * **noise sigma anywhere in model math** -> error. A sigma has no legitimate mechanistic role,
      and reusing one only ever buys a looser noise model — the half of the Brannmark failure that
      masks a wrong mechanism.
    * **observable-side symbol in a rate law that declares NO parameter of its own** -> error. The
      symbol is standing in for that reaction's rate constant, which is exactly the Brannmark cheat
      (``IR => IRp; cell * IR * k_IRP_1Step``) and exactly what "dodging a real kinetic parameter"
      means.
    * **anything else** -> warning. A legitimate reuse always sits ALONGSIDE a dedicated rate
      constant (``CISRNAEqc * CISRNATurn * npSTAT5 * ActD / init_STAT5``), or is confined to an
      initial assignment or rule.
    """
    sm = problem.model.sbml_model
    obs_symbols = observation_model_symbols(problem)
    in_math = (obs_symbols & parameter_ids(sm)) & _model_math_names(sm)
    sigmas = in_math & noise_sigma_ids(problem)

    # A reaction whose rate law uses an observation-model parameter but declares NO dedicated
    # parameter of its own is the Brannmark cheat: the observation symbol is standing in for the
    # rate constant. A legitimate reuse always sits ALONGSIDE a real kinetic parameter — every
    # bachmann rate law that normalises by an abundance still has its own turnover/activation
    # constant (`CISRNAEqc * CISRNATurn * npSTAT5 * ActD / init_STAT5`; `Epo * EpoRJAK2 *
    # JAK2ActEpo / (SOCS3 * SOCS3Inh / SOCS3Eqc + 1)`).
    # Hoisted: parameter_ids walks every SBML parameter, and validate_problem runs on every edit.
    all_params = parameter_ids(sm)
    # A condition target is an experimental INPUT (a dose, a switch), not a rate constant the model
    # supplies, so it must not count as the reaction's "own" parameter — otherwise the Brannmark
    # cheat escapes simply by multiplying the dose target in:
    # `IR => IRp; cell*IR*k_IRP_1Step*insulin_level`.
    own_params = all_params - obs_symbols - condition_target_symbols(problem)
    rate_constant_role: set[str] = set()
    for i in range(sm.getNumReactions()):
        kl = sm.getReaction(i).getKineticLaw()
        if kl is None:
            continue
        names = _ast_names(kl.getMath())
        used_obs = names & in_math
        if not used_obs:
            continue
        if not names & own_params:
            rate_constant_role |= used_obs
    issues = [
        {
            "level": "error",
            "message": (
                f"'{pid}' is a NOISE SIGMA (resolved through an observable's noiseFormula or the "
                "measurement table's noiseParameters) but is ALSO used inside a reaction, rule, or "
                "initial assignment. Reusing a sigma as a mechanistic quantity lets a wrong "
                "mechanism hide behind an artificially loose noise model instead of the sigma "
                "being estimated as what it actually is. Introduce a SEPARATE, dedicated "
                "parameter for the reaction instead."
            ),
            "check": "observation_model_reuse",
        }
        for pid in sorted(sigmas)
    ]
    issues += [
        {
            "level": "error",
            "message": (
                f"'{pid}' is an observation-model parameter (an observable scaling/offset) and is "
                "used in a reaction whose rate law declares NO parameter of its own — so it is "
                "acting AS that reaction's rate constant. Reusing an observation parameter this way "
                "lets the model dodge introducing a real kinetic parameter. Normalising by such a "
                "quantity alongside a dedicated rate constant is fine; standing in for one is not. "
                "Introduce a SEPARATE, dedicated parameter for the reaction."
            ),
            "check": "observation_model_reuse",
        }
        for pid in sorted(rate_constant_role - sigmas)
    ]
    issues += [
        {
            "level": "warning",
            "message": (
                f"'{pid}' is an observable-scaling/offset parameter that is also used inside a "
                "reaction, rule, or initial assignment. That is legitimate when it is genuinely "
                "the same physical quantity on both sides (a total abundance that sets an initial "
                "condition and normalises a relative readout). It is NOT legitimate as a way to "
                "avoid introducing a real kinetic parameter — check that this reuse is the former."
            ),
            "check": "observation_model_reuse",
        }
        for pid in sorted(in_math - sigmas - rate_constant_role)
    ]
    return issues


def _fixed_constant_role_issues(problem, protected_fixed_ids) -> list[dict[str, Any]]:
    """Hard-error when a parameter that ships FIXED and is a mechanistic/structural constant —
    a conservation total, a known rate, a measured concentration — has been REMOVED from the
    model or turned ESTIMATED.

    ``protected_fixed_ids`` is the caller-computed set of such constants (see
    :meth:`session.PetabSession.protected_fixed_parameter_ids`); it already excludes
    initial-condition parameters (removed legitimately with their species) and observation-model
    factors (their own estimation paths). A published problem fixes these for a reason, and:

    * **removing** one drops a piece of the problem's structure (an orphaned conservation total
      or known rate silently pruned when its reaction is dropped/re-authored);
    * **estimating** one lets its value DRIFT off the constant (a benchmark run let three
      conservation totals fixed at 1 wander to 9 / 0.001 / 0.4) and adds to k, inflating BIC.

    Nothing toggles a parameter's estimated/fixed status directly (there is no such tool), and a
    shipped-fixed parameter keeps that status only while it stays declared — so the recourse for
    both is the same: keep the constant declared in the model, as shipped. Passing an empty/None
    set (e.g. a problem with no such constants, or an unreadable v0) yields no issues.
    """
    if not protected_fixed_ids:
        return []
    pdf = getattr(problem, "parameter_df", None)
    if pdf is None:
        return []
    current = {str(pid) for pid in pdf.index}

    def _estimated(pid) -> bool:
        try:
            return ESTIMATE in pdf.columns and as_bool(pdf.loc[pid, ESTIMATE])
        except (TypeError, ValueError, KeyError):
            return False

    issues: list[dict[str, Any]] = []
    for pid in sorted(protected_fixed_ids):
        if pid not in current:
            issues.append(
                {
                    "level": "error",
                    "message": (
                        f"'{pid}' ships FIXED (estimate=0) in this problem — a known constant (a "
                        "conservation total, a measured concentration, a fixed rate) — but this edit "
                        "REMOVED it from the model. A shipped-fixed constant is part of the problem's "
                        "structure; dropping it changes the problem. Keep it declared AND used: a "
                        "shipped-fixed constant must be referenced in a rate law, rule, or initial "
                        "assignment (a constant you declare but never use is pruned as an orphan, "
                        "which trips this too). Retain the reaction/expression that uses it."
                    ),
                    "check": "fixed_constant_removed",
                }
            )
        elif _estimated(pid):
            issues.append(
                {
                    "level": "error",
                    "message": (
                        f"'{pid}' ships FIXED (estimate=0) in this problem but is now an ESTIMATED "
                        "parameter. A parameter the problem fixes is a known constant — a conservation "
                        "total, a measured concentration, a fixed rate — and estimating it lets its "
                        "value drift off that constant while adding to k (the estimated-parameter "
                        "count), which inflates BIC. There is no tool to toggle a parameter between "
                        "fixed and estimated; a shipped-fixed constant is re-added ESTIMATED only if "
                        "it was dropped and its symbol later reintroduced. Keep the constant declared "
                        "in the model, as shipped, so its fixed status is preserved."
                    ),
                    "check": "fixed_constant_made_estimated",
                }
            )
    return issues


def _undefined_readout_issues(problem) -> list[dict[str, Any]]:
    """One error per ``readout_*`` symbol an observable formula references but the model leaves
    undefined — with the remedy and the observable's own biological description in the message.

    ``readout_<observableId>`` is the blinded-benchmark measurement-mapping contract
    (``petab_harness.benchmark._blind_observables``): the shipped observable formula references a
    symbol the MODEL must define via an assignment rule, so what the assay measures is stated
    (``observableName``) but how it maps onto the model's states is the modeller's to decide.
    petab's own lint already errors on the missing entity, but generically — nothing tells the
    agent the fix is one ``set_assignment`` op, and nothing repeats what the readout measures at
    the moment that information is needed. Burned turns follow. This check owns the remedy.

    A model with no dynamics yet (the v0 seed) declares every readout a constant, so this fires
    only once the agent has authored a model that forgot a mapping. Best-effort:
    any failure returns ``[]`` rather than breaking validation.
    """
    from petab.v2.C import OBSERVABLE_FORMULA, OBSERVABLE_NAME

    try:
        from .reconcile import formula_identifiers

        odf = getattr(problem, "observable_df", None)
        sm = getattr(problem.model, "sbml_model", None)
        if odf is None or sm is None or OBSERVABLE_FORMULA not in odf.columns:
            return []
        # The SEED declares every observable-formula identifier so v0 lints and simulates, and
        # that includes every readout_*. So "the id resolves to some SBML element"
        # is satisfied by construction and cannot be the test -- checking it made this whole guard
        # a no-op, which is how a model leaving every readout a FREE CONSTANT would have committed:
        # on blasi that is 16 constants against 16 steady-state observable means, fitting them
        # exactly while skipping the measurement-mapping task entirely.
        #
        # The test is therefore an ASSIGNMENT RULE, and the seed is skipped by the same
        # predicate `_unused_condition_target_issues` uses -- `has_dynamics`, NOT a bare reaction
        # count. A model can be fully dynamic with zero reactions (rate rules are the other half
        # of `has_dynamics`, and ODE-style authoring is a normal Antimony idiom), and counting
        # reactions alone let exactly those models through the gate with every readout left as a
        # free constant -- the loophole this check exists to close.
        if not has_dynamics(sm):
            return []
        rule_targets = {
            sm.getRule(i).getVariable()
            for i in range(sm.getNumRules())
            if sm.getRule(i).isAssignment()
        }
        issues: list[dict[str, Any]] = []
        # One issue per READOUT SYMBOL, not per observable. isensee references readout_pRII from
        # 117 of its 123 rows, so per-row emission meant ~117 byte-identical errors (~100kB with
        # its description) on a single rejected commit — the payload cap `_ID_LIST_CAP` exists to
        # prevent elsewhere.
        seen_readouts: set[str] = set()
        for obs_id, row in odf.iterrows():
            formula = str(row[OBSERVABLE_FORMULA])
            if "readout_" not in formula:  # skip the L3 parse on unblinded rows
                continue
            for name in sorted(formula_identifiers(formula)):
                if not name.startswith("readout_") or name in seen_readouts:
                    continue
                seen_readouts.add(name)
                if name in rule_targets:
                    seen_readouts.discard(name)  # defined: do not consume the slot
                    continue
                # F9: a present-but-empty observableName cell is float NaN, which is TRUTHY --
                # `or ""` does not catch it and the agent is told "It measures: nan."
                raw = row.get(OBSERVABLE_NAME, "")
                measures = raw.strip() if isinstance(raw, str) else ""
                measured = f" It measures: {measures}." if measures else ""
                issues.append(
                    {
                        "level": "error",
                        "message": (
                            f"observable {obs_id!r} references {name!r}, a MEASUREMENT MAPPING your "
                            f"model must define — say what this readout is in terms of YOUR model's "
                            f"states with an ASSIGNMENT RULE: `{name} := <expression of your "
                            f"species>` (op set_assignment). Declaring it as a species or a "
                            f"parameter does NOT define it: a free constant is not a "
                            f"measurement.{measured}"
                        ),
                        "check": "readout_mapping",
                    }
                )
        return issues
    except Exception:  # pragma: no cover - advisory check must never break validation
        return []


def _unused_condition_target_issues(problem) -> list[dict[str, Any]]:
    """Hard-error when the model declares a condition-table target but never USES it.

    Each condition-table target sets a model quantity per condition — a dose, an input level, a
    genotype switch, a rate multiplier: the very thing that makes the conditions differ. petab
    requires the target to be a model entity (a species or parameter) and reconcile keeps it (a
    condition target is never pruned), so a target the model declares but never references in a rate
    law, rule, initial assignment, or reaction — nor in the observation model — passes the lint yet
    does NOTHING: every condition then simulates identically and the experimental design is silently
    ignored. A run can then "fit" by tuning other parameters while the perturbation it should be
    responding to is inert. Gate it here so the model actually consumes each condition-table
    quantity.

    Only targets that ARE model entities are checked (a target absent from the model is a
    different error petab lint already reports). Best-effort: no condition table / no model
    yields no issues.

    A model with NO DYNAMICS is skipped entirely. The question here — "does the authored
    mechanism consume each condition-table quantity?" — presupposes a mechanism; against a model
    that cannot move, every target is trivially unused and the check would reject the
    content-free seed (see :func:`reconcile.seed_model_antimony`) and, worse, every later
    parameters-only edit made while the seed is still the working model. This cannot become a
    loophole: ``make_changes`` refuses to commit a model without dynamics, using this very
    predicate, so anything a caller actually authors is checked.
    """
    cdf = getattr(problem, "condition_df", None)
    model = getattr(problem, "model", None)
    sm = model.sbml_model if model is not None else None
    if cdf is None or sm is None or not has_dynamics(sm):
        return []

    used = _model_math_names(sm) | _reaction_species(sm) | observation_model_symbols(problem)
    model_entities = parameter_ids(sm) | species_ids(sm)
    targets = condition_target_symbols(problem)
    # A target with a SINGLE distinct value across every condition encodes no experimental design:
    # it cannot make one condition differ from another, so "the conditions don't perturb the model"
    # simply does not apply and requiring the model to consume it is wrong. Published exports carry
    # such targets — raimundez's METinh_level is 0 in all 174 conditions because the paper used MET
    # inhibition for a PREDICTION validated against separate data, not for calibration. Forcing the
    # agent to wire one in buys a submodule that can never carry flux. petab lint still requires the
    # target to EXIST as a model entity; this only drops the requirement that it be USED. A
    # condition that leaves the target unset counts as one more distinct value, as an empty cell
    # did in the wide v1 table.
    # Requires MORE THAN ONE condition: with a single condition every target is trivially
    # constant, yet the condition is still the only thing setting that value, so the model should
    # consume it. The exemption is about a target that cannot DISTINGUISH conditions.
    n_conditions = cdf[CONDITION_ID].nunique()
    if n_conditions > 1:
        for tgt, sub in cdf.groupby(TARGET_ID, sort=False):
            values = sub[TARGET_VALUE].astype(str).nunique()
            if values + (sub[CONDITION_ID].nunique() < n_conditions) <= 1:
                targets.discard(str(tgt))

    return [
        {
            "level": "error",
            "message": (
                f"The condition table sets '{tgt}' per condition, but the model never uses "
                f"'{tgt}' — it appears in no rate law, rule, initial assignment, reaction, or "
                "observation formula. The conditions therefore don't perturb the model: every "
                "condition simulates identically and the experimental design that distinguishes "
                f"them is ignored. Use '{tgt}' in the mechanism — as an input/driver in a rate "
                "law, or as the initial state of a species that participates in the dynamics."
            ),
            "check": "unused_condition_target",
        }
        for tgt in sorted(targets)
        if tgt in model_entities and tgt not in used
    ]


# -- authoring-time check: frozen parametric reaction stoichiometry (checks the CONVERTED SBML) --
def frozen_parametric_stoichiometry_issues(sbml_model) -> list[dict[str, Any]]:
    """Hard-error when a reaction stoichiometry was written as a PARAMETER/symbol in Antimony.

    When an Antimony reaction uses a symbol as a stoichiometry coefficient — e.g.
    ``Naive -> N_burst EarlyEffector`` with ``N_burst`` a declared parameter — libantimony's SBML
    export FREEZES the symbol's nominal value into a constant literal stoichiometry and DROPS the
    parameter: the intended estimable amplification factor silently becomes a fixed constant that
    never enters the fit, with no error (issue #89). It gets worse when the symbol is also used in
    the rate law — it is dropped there too, leaving a dangling reference.

    libantimony leaves exactly one fingerprint of the frozen symbol: it sets the
    ``SpeciesReference`` **id** to the original symbol name. A literal coefficient (``-> 2 X``)
    leaves the id UNSET, so the two cases — which carry an identical numeric ``stoichiometry`` —
    are cleanly distinguishable on the converted SBML (verified against libantimony 3.2). This is
    why the check reads the CONVERTED model rather than trying to re-parse the Antimony source.

    Reusing ONE symbol across several stoichiometries (``N A -> N B``) freezes and drops it exactly
    the same way, but only the FIRST occurrence keeps the fingerprint: the rest become
    ``<reaction>_<species>_stoich`` species references driven by an assignment rule that reads the
    now-undeclared symbol. One flagged reference is enough to reject the model, so the extra
    shape needs no separate detection.

    Scope: this is an Antimony-CONVERSION artifact, so it is run on the freshly-converted candidate
    at the ``set_model`` authoring boundary (see :func:`operations.set_model`), NOT wired into
    :func:`validate_problem`. A hand-authored or loaded SBML L3 model may legitimately give a
    constant species reference an id, which is not this bug — so this must never run against an
    arbitrary loaded model.

    A species reference is flagged only when it is CONSTANT (a frozen coefficient) and its id is
    NOT the target of an initial assignment or rule (a rule-driven, genuinely variable
    stoichiometry is not frozen and would be legitimately estimable). Returns the standard
    ``{level, message, check}`` issue dicts, deduplicated by ``(reaction, species, symbol)``.
    Best-effort: any failure yields ``[]`` — a detection bug must never hard-block authoring.
    """
    try:
        return _frozen_parametric_stoichiometry_issues(sbml_model)
    except Exception:  # noqa: BLE001 - never let a detection bug block authoring
        return []


def _frozen_parametric_stoichiometry_issues(sbml_model) -> list[dict[str, Any]]:
    # A species-reference id targeted by an initial assignment or a rule is a genuinely variable
    # (rule-driven) stoichiometry, not a frozen constant — never flag those.
    driven: set[str] = set()
    for i in range(sbml_model.getNumInitialAssignments()):
        driven.add(sbml_model.getInitialAssignment(i).getSymbol())
    for i in range(sbml_model.getNumRules()):
        rule = sbml_model.getRule(i)
        if rule.isSetVariable():
            driven.add(rule.getVariable())

    issues: list[dict[str, Any]] = []
    seen: set[tuple[str | None, str, str]] = set()
    for i in range(sbml_model.getNumReactions()):
        reaction = sbml_model.getReaction(i)
        rid = reaction.getId() or None
        sides = (
            ("reactant", reaction.getReactant, reaction.getNumReactants()),
            ("product", reaction.getProduct, reaction.getNumProducts()),
        )
        for side, get_ref, count in sides:
            for j in range(count):
                sref = get_ref(j)
                # Frozen-symbol fingerprint: a CONSTANT species reference carrying an id that no
                # rule drives. A literal coefficient leaves the id unset and is skipped here.
                if not sref.isSetId() or not sref.getConstant():
                    continue
                symbol = sref.getId()
                if symbol in driven:
                    continue
                species = sref.getSpecies()
                key = (rid, species, symbol)
                if key in seen:
                    continue
                seen.add(key)
                issues.append(
                    {
                        "level": "error",
                        "message": _frozen_stoichiometry_message(
                            rid, side, species, symbol, sref.getStoichiometry()
                        ),
                        "check": "symbolic_stoichiometry",
                    }
                )
    return issues


def _frozen_stoichiometry_message(rid, side: str, species: str, symbol: str, frozen: float) -> str:
    where = f"Reaction '{rid}'" if rid else f"A reaction (on its {side} side)"
    # Tailor the estimable-case guidance to the side: a PRODUCT coefficient amplifies the product's
    # flux; a REACTANT coefficient scales that reactant's own consumption. A reaction's single rate
    # drives every participant, so the factor has to move into a per-species rate term (splitting
    # the reaction if needed) rather than multiplying the whole reaction rate.
    if side == "product":
        split = (
            f"split it into a unit-stoichiometry consumption '<reactant> -> ; <rate>' and a scaled "
            f"production '-> {species}; {symbol} * <rate>', so only '{species}' is amplified while "
            "the reactant is still consumed 1:1"
        )
    else:
        split = (
            f"scale only that reactant's own consumption — e.g. '{species} -> ; {symbol} * <rate>' "
            f"with the product formed in its own reaction — so only '{species}' is consumed "
            f"'{symbol}'-fold"
        )
    return (
        f"{where} uses parameter/symbol '{symbol}' as the stoichiometry coefficient of species "
        f"'{species}'. Antimony froze it to the constant {frozen:g} and DROPPED '{symbol}' from "
        "the model, so it is neither estimable nor present — the intended tunable factor silently "
        "became a fixed constant. If the coefficient is a known numeric constant, write that number "
        f"directly as a literal (e.g. '{frozen:g} {species}' or '2.0 {species}'), which is preserved correctly. If it must be "
        "ESTIMATED it cannot be a stoichiometry at all — a reaction's single rate drives every "
        f"participant, so {split}, keeping '{symbol}' a real estimated parameter in the rate law."
    )


def validate_problem(problem, protected_fixed_ids=None) -> dict[str, Any]:
    """Run PEtab v2 linting + libsbml SBML consistency and return a JSON-friendly report.

    Returns ``valid`` (no errors from either check), ``n_errors``/``n_warnings``, and the
    captured ``issues`` (level/message/check) — petab table issues first, then SBML ones.

    ``protected_fixed_ids`` (the shipped-fixed mechanistic/structural constants, from
    :meth:`session.PetabSession.protected_fixed_parameter_ids`) enables the role check that
    rejects removing or estimating one (see :func:`_fixed_constant_role_issues`); omit it to
    skip that check. The condition-target check (see
    :func:`_unused_condition_target_issues`) always runs — it needs no reference, and it
    self-skips the one case it cannot answer (a model with no dynamics).
    """
    lint = [
        i
        for i in problem.to_petab().validate()
        if i.level >= ValidationIssueSeverity.ERROR
        or (i.level == ValidationIssueSeverity.WARNING and i.task not in _DESIGN_ONLY_CHECKS)
    ]
    issues = [
        {"level": i.level.name.lower(), "message": i.message, "check": i.task or "petab"}
        for i in lint
    ]
    has_errors = any(i.level >= ValidationIssueSeverity.ERROR for i in lint)
    n_errors = sum(1 for i in lint if i.level >= ValidationIssueSeverity.ERROR)
    n_warnings = sum(1 for i in lint if i.level == ValidationIssueSeverity.WARNING)

    sbml_issues = _sbml_consistency_issues(problem)
    issues += sbml_issues
    sbml_errors = sum(1 for i in sbml_issues if i["level"] == "error")
    n_errors += sbml_errors
    n_warnings += sum(1 for i in sbml_issues if i["level"] == "warning")

    reuse_issues = _observation_model_reuse_issues(problem)
    issues += reuse_issues
    # Only the noise-sigma half is an error; observable-side reuse is a legitimate warning.
    reuse_errors = [i for i in reuse_issues if i["level"] == "error"]
    n_errors += len(reuse_errors)
    n_warnings += sum(1 for i in reuse_issues if i["level"] == "warning")

    role_issues = _fixed_constant_role_issues(problem, protected_fixed_ids)
    issues += role_issues
    n_errors += len(role_issues)

    condition_issues = _unused_condition_target_issues(problem)
    issues += condition_issues
    n_errors += len(condition_issues)

    readout_issues = _undefined_readout_issues(problem)
    issues += readout_issues
    n_errors += len(readout_issues)

    return {
        "valid": (
            (not has_errors)
            and sbml_errors == 0
            and not reuse_errors
            and not role_issues
            and not condition_issues
            and not readout_issues
        ),
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "issues": issues,
    }
