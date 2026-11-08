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

import libsbml
from petab.v2.C import ESTIMATE, NOMINAL_VALUE, TARGET_ID, TARGET_VALUE
from petab.v2.lint import ValidationIssueSeverity

from .problem import as_bool, later_changes
from .reconcile import (
    _ast_names,
    _declared_initial,
    _observable_and_measurement_symbols,
    condition_target_symbols,
    formula_symbols,
    input_species,
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
# condition or experiment no measurement uses — a problem may ship its held-out conditions in the
# condition table), which no call can change. Promoting them would put a fixed, unactionable
# string on the `warnings` of EVERY make_changes result, into a history nothing prunes, so they are
# dropped at the capture site and reach neither `issues` nor the warning COUNT. Warnings only: an
# ERROR from any check is kept.
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

    This was caught in practice: two independent agents wired a problem's four fixed
    observable-scaling factors directly into their reaction rate laws instead of introducing the
    ~14 real kinetic parameters the problem expects, ending up with a severely under-mechanistic
    model that still passed its own chi2 goodness-of-fit check (with help from noise sigmas
    inflated 20-60x). Gate this at commit time rather than relying
    on an agent to notice: reusing one of these symbols does not "cost" anything on its own
    metrics, so nothing else in the pipeline would catch it.

    Blanket rejection is wrong, because a published problem legitimately reuses these symbols: the
    SAME physical quantity is often both a model quantity and the observable's normaliser. The
    reference case is a signalling model whose total abundances (``A_tot``, ``B_tot``, ``C_tot``)
    set the initial conditions AND divide the relative pA/pB/C observables, and whose equilibrium
    constants (``K1``, ``K2``, ``K3``, ``K4``) scale both the species and their readouts. Making
    that an error rejects the published mechanism itself, and with it any agent that finds it.

    Restricting the exemption to INITIAL ASSIGNMENTS does not work either: all seven of those
    symbols also appear in rate laws, and two of them (``K3``, ``K4``) have no initial assignment
    at all. Nor does "denominator only" — ``K1`` and ``K3`` appear as numerator factors. What
    actually separates the two cases is the symbol's ROLE:

    * **noise sigma anywhere in model math** -> error. A sigma has no legitimate mechanistic role,
      and reusing one only ever buys a looser noise model — the half of the scaling-factor failure
      that masks a wrong mechanism.
    * **observable-side symbol in a rate law that declares NO parameter of its own** -> error. The
      symbol is standing in for that reaction's rate constant, which is exactly the scaling-factor
      cheat (``A => pA; cell * A * s_A``) and exactly what "dodging a real kinetic parameter"
      means.
    * **anything else** -> warning. A legitimate reuse always sits ALONGSIDE a dedicated rate
      constant (``K3 * k_turn * pB * D / B_tot``), or is confined to an initial assignment or
      rule.
    """
    sm = problem.model.sbml_model
    obs_symbols = observation_model_symbols(problem)
    in_math = (obs_symbols & parameter_ids(sm)) & _model_math_names(sm)
    sigmas = in_math & noise_sigma_ids(problem)

    # A reaction whose rate law uses an observation-model parameter but declares NO dedicated
    # parameter of its own is the scaling-factor cheat: the observation symbol is standing in for
    # the rate constant. A legitimate reuse always sits ALONGSIDE a real kinetic parameter — every
    # rate law of the reference case that normalises by an abundance still has its own
    # turnover/activation constant (`K3 * k_turn * pB * D / B_tot`; `L * R * k_act /
    # (C * k_inh / K2 + 1)`).
    # Hoisted: parameter_ids walks every SBML parameter, and validate_problem runs on every edit.
    all_params = parameter_ids(sm)
    # A condition target is an experimental INPUT (a dose, a switch), not a rate constant the model
    # supplies, so it must not count as the reaction's "own" parameter — otherwise the
    # scaling-factor cheat escapes simply by multiplying the dose target in:
    # `A => pA; cell*A*s_A*dose`.
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
    """Warn when a parameter that ships FIXED and is a mechanistic/structural constant — a
    conservation total, a known rate, a measured concentration — is no longer read by the model,
    and hard-error when it has been turned ESTIMATED.

    ``protected_fixed_ids`` is the caller-computed set of such constants (see
    :meth:`session.PetabSession.protected_fixed_parameter_ids`); it already excludes
    initial-condition parameters (removed legitimately with their species) and observation-model
    factors (their own estimation paths). A published problem fixes these for a reason, and:

    * **removing** one drops a piece of the problem's structure (an orphaned conservation total
      or known rate silently pruned when its reaction is dropped/re-authored);
    * **estimating** one lets its value DRIFT off the constant (one fit let three
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
    removed = [pid for pid in sorted(protected_fixed_ids) if pid not in current]
    if removed:
        # A WARNING, one for every constant. Refused, it made agents that did not know what a
        # constant meant fake a use of it (`0*k_const`, a readout of constants only),
        # which BIC cannot see; unread, the constant costs nothing and whatever the model
        # estimates in its place costs k. A constant read again comes back fixed, from the
        # shipped row (`reconcile.sync_parameter_table`).
        issues.append(
            {
                "level": "warning",
                "message": (
                    f"shipped FIXED constant(s) {removed} are not read by the model. They are "
                    "known constants of the experiment (a conservation total, a measured dose, a "
                    "fixed rate): a model that reads them, as the quantity each is, estimates "
                    "nothing in their place."
                ),
                "check": "fixed_constant_removed",
            }
        )
    for pid in sorted(protected_fixed_ids):
        if pid in removed:
            continue
        if _estimated(pid):
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


def _undefined_readout_issues(problem, statements: bool = False) -> list[dict[str, Any]]:
    """One error per ``readout_*`` symbol an observable formula references but the model leaves
    undefined — with the remedy and the observable's own biological description in the message.

    ``readout_<observableId>`` is the measurement-mapping contract of a problem that ships its
    observables as ``readout_*`` mappings: the shipped observable formula references a
    symbol the MODEL must define via an assignment rule, so what the assay measures is stated
    (``observableName``) but how it maps onto the model's states is the modeller's to decide.
    petab's own lint already errors on the missing entity, but generically — nothing tells the
    agent the fix is one ``set_assignment`` op, and nothing repeats what the readout measures at
    the moment that information is needed. Burned turns follow. This check owns the remedy.

    A model with no dynamics yet (the v0 seed) declares every readout a constant, so this fires
    only once the agent has authored a model that forgot a mapping. Best-effort:
    any failure returns ``[]`` rather than breaking validation.

    ``statements``: the model came from make_changes' ``statements`` route, where a readout is
    given in ``readouts``; the assignment-rule remedy (``set_assignment``) is the Antimony routes'
    and is not offered there.
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
        # on one problem that is 16 constants against 16 steady-state observable means, fitting them
        # exactly while skipping the measurement-mapping task entirely.
        #
        # The test is therefore an ASSIGNMENT RULE, and the seed is skipped by `has_dynamics`,
        # NOT a bare reaction count. A model can be fully dynamic with zero reactions (rate rules are the other half
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
        # One issue per READOUT SYMBOL, not per observable. One problem references one readout
        # from 117 of its 123 rows, so per-row emission meant ~117 byte-identical errors (~100kB
        # with its description) on a single rejected commit — the payload cap `_ID_LIST_CAP`
        # exists to prevent elsewhere.
        seen_readouts: set[str] = set()
        for obs_id, row in odf.iterrows():
            formula = str(row[OBSERVABLE_FORMULA])
            if "readout_" not in formula:  # skip the L3 parse on rows with no readout
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
                how = (
                    f'in `readouts`, a formula over the amounts it reads (`"{name}": '
                    f'"amount(<agent pattern>)"`, or over names declared in `amounts`)'
                    if statements
                    else f"with an ASSIGNMENT RULE: `{name} := <expression of your species>` (op "
                    f"set_assignment)"
                )
                issues.append(
                    {
                        "level": "error",
                        "message": (
                            f"observable {obs_id!r} references {name!r}, a MEASUREMENT MAPPING your "
                            f"model must define — say what this readout is in terms of YOUR model's "
                            f"states {how}. Declaring it as a species or a parameter does NOT "
                            f"define it: a free constant is not a measurement.{measured}"
                        ),
                        "check": "readout_mapping",
                    }
                )
        return issues
    except Exception:  # pragma: no cover - advisory check must never break validation
        return []


#: How a condition target enters a model so that every period's value reaches it: the remedy of
#: :func:`_first_period_only_issues`, and the end of :data:`INPUT_UNREAD`.
INPUT_REMEDY = (
    "read it as a parameter of its id in a rate law, which follows every period's value, or name "
    "a species by its id, which every period sets (on the statements route: a row's binding "
    '`kf="k*dose"`, or a seed of the agent it stands for at the id alone, '
    '`{"agent": "FSK", "amount": "Fsk"}`). An initial amount that reads it (a seed `2*dose`) '
    "takes the first period's value alone, which in a pre-equilibrated experiment is the "
    "pre-equilibration one."
)

#: What a model may do with a condition target it declares, and :data:`INPUT_REMEDY` for one it
#: means to read: the end of :func:`_unread_input_issues`' warning and of the invalid-targets hint
#: (:mod:`operations`), which agents read as "every input must be read" (PEtab-MCP#290).
INPUT_UNREAD = (
    "Leaving an input unread is allowed, and right for one with no effect on the data, such as a "
    "control: a term for it is a parameter the score charges. To give one an effect, "
    + INPUT_REMEDY
)

#: Periods named per target in :func:`_first_period_only_issues`; the rest are counted.
_PERIODS_SHOWN = 3


def _first_period_only_issues(problem) -> list[dict[str, Any]]:
    """One warning naming each condition target that a period after an experiment's first sets
    while the model reads it only in initial assignments, with the periods: those see the first
    period's value alone (:func:`problem.later_changes`), so the later one never reaches the model.
    libpetab's unread-target check passes it, as the target is read."""
    sm = getattr(problem.model, "sbml_model", None)
    later = later_changes(problem) if sm is not None else {}
    if not later:
        return []
    at_start = set()
    for assignment in sm.getListOfInitialAssignments():
        at_start |= _ast_names(assignment.getMath())
    read = _model_math_names(sm, initial=False) | _observable_and_measurement_symbols(problem)
    stale = sorted(t for t in later if t in at_start and t not in read)
    if not stale:
        return []

    def periods(target: str) -> str:
        shown = ", ".join(f"{e} at t = {t:g}" for e, t in later[target][:_PERIODS_SHOWN])
        more = len(later[target]) - _PERIODS_SHOWN
        return f"`{target}` ({shown}{f' and {more} more' if more > 0 else ''})"

    return [
        {
            "level": "warning",
            "message": (
                f"condition target(s) {'; '.join(map(periods, stale))} change in these periods, "
                "after their experiment's first, but are read only in initial amounts, so the "
                f"model never sees the new value. For each, {INPUT_REMEDY}"
            ),
            "check": "input_read_only_at_start",
        }
    ]


def _unread_input_issues(problem) -> list[dict[str, Any]]:
    """One warning naming each condition target the model declares and nothing reads: no rate,
    event, initial amount or observable, directly or through the assignment rules they read, so no
    period's value reaches the model. libpetab's invalid-target check passes it, as the target is
    declared; the reconcile leaves it so when it prunes the seed whose initial amount read it
    (PEtab-MCP#205), and an edit when it drops the last reader of a rule that reads it
    (PEtab-MCP#264)."""
    sm = getattr(problem.model, "sbml_model", None)
    if sm is None:
        return []
    declared = parameter_ids(sm) | species_ids(sm)
    rules = {r.getVariable(): _ast_names(r.getMath()) for r in sm.getListOfRules() if r.isAssignment()}
    read = (
        _model_math_names(sm, assignments=False)
        | _reaction_species(sm)
        | _observable_and_measurement_symbols(problem)
    )
    while reached := read & rules.keys():
        for target in reached:
            read |= rules.pop(target)
    unread = sorted((condition_target_symbols(problem) & declared) - read)
    if not unread:
        return []
    return [
        {
            "level": "warning",
            "message": (
                f"condition target(s) {', '.join(f'`{t}`' for t in unread)} are read by nothing "
                "in the model, or only by assignment rules nothing reads, so the model never sees "
                f"their value. {INPUT_UNREAD}"
            ),
            "check": "input_unread",
        }
    ]


def _sourceless_species_issues(problem) -> list[dict[str, Any]]:
    """One warning per species a rate law or rule reads that stays 0: it starts at 0, nothing sets
    it (an input, an event, a condition) to a value that is not 0, and no reaction that can run
    changes it. ``set_rate_law`` replaces a reaction's rate alone, so one sent to give a reaction a
    new product left that species with no source: the rate laws reading it read 0, their
    parameters pinned at a bound, and the agent took that for the data rejecting the mechanism
    (PEtab-MCP#270).

    "Can run" is decided all together: a reaction runs unless its rate law is 0 whenever every
    species still held at 0 and every parameter the table fixes at 0 is 0
    (:func:`_vanishes_at`), and each species such a reaction changes is let go, until none is."""
    sm = getattr(problem.model, "sbml_model", None)
    if sm is None:
        return []
    ruled = {r.getVariable(): r.getMath() for r in sm.getListOfRules() if r.isAssignment()}
    zero = _fixed_zero_parameters(problem)

    def vanishes(math) -> bool:
        return math is not None and _vanishes_at(math, zero, ruled)

    cdf = problem.condition_df
    conditions = (
        zip(cdf[TARGET_ID], cdf[TARGET_VALUE])
        if cdf is not None and {TARGET_ID, TARGET_VALUE} <= set(cdf.columns)
        else ()
    )
    held = (
        input_species(sm)
        | {str(target) for target, value in conditions if not _is_zero(value)}
        | {
            ea.getVariable()
            for event in sm.getListOfEvents()
            for ea in event.getListOfEventAssignments()
            if not vanishes(ea.getMath())
        }
    )

    def starts_at_zero(sp) -> bool:
        assignment = sm.getInitialAssignmentBySymbol(sp.getId())
        if assignment is not None:
            return vanishes(assignment.getMath())
        value, _ = _declared_initial(sm, sp)
        return value is None or value == 0

    zero |= {
        sp.getId() for sp in sm.getListOfSpecies() if sp.getId() not in held and starts_at_zero(sp)
    }
    reactions = []
    for rxn in sm.getListOfReactions():
        net: dict[str, float] = {}
        for sign, refs in ((-1, rxn.getListOfReactants()), (1, rxn.getListOfProducts())):
            for ref in refs:
                net[ref.getSpecies()] = net.get(ref.getSpecies(), 0) + sign * ref.getStoichiometry()
        law = rxn.getKineticLaw()
        reactions.append((law.getMath() if law is not None else None, net))
    while (
        changed := {
            sid for math, net in reactions if not vanishes(math) for sid, n in net.items() if n != 0
        }
        & zero
    ):
        zero -= changed

    readers: dict[str, list[str]] = {}
    for rxn, (math, _) in zip(sm.getListOfReactions(), reactions):
        for sid in _ast_names(math) & zero:
            readers.setdefault(sid, []).append(f"`{rxn.getId()}`")
    for rule in sm.getListOfRules():
        for sid in _ast_names(rule.getMath()) & zero:
            readers.setdefault(sid, []).append(f"the rule for `{rule.getVariable()}`")
    return [
        {
            "level": "warning",
            "message": (
                f"species `{sid}` starts at 0 and stays 0: no reaction produces it, and no "
                f"condition, event or rule sets it. The rate laws and rules that read it "
                f"({', '.join(readers[sid])}) see 0 throughout, so a parameter acting only "
                "through it cannot change the fit. Give it a source: a reaction that produces "
                "it, or a nonzero initial amount."
            ),
            "check": "species_without_source",
        }
        for sid in sorted(readers.keys() & species_ids(sm))
    ]


def _constant_noise_species_issues(problem) -> list[dict[str, Any]]:
    """One error per species a noise formula reads that no reaction changes (it takes part in none,
    or with as much on each side) and no rule sets. AMICI holds such a species as a constant total
    and refuses at import to build a noise model that reads one ("the sigma of obs depends on
    tcl_B"): armistead 21 met it as an ill-posed model and changed three unrelated things
    (PEtab-MCP#285). A condition target or a modifier in no reaction is fine where no noise reads
    it."""
    from petab.v2.C import NOISE_FORMULA, NOISE_PARAMETERS, OBSERVABLE_ID

    sm = getattr(problem.model, "sbml_model", None)
    if sm is None or not has_dynamics(sm):
        return []
    changed = {r.getVariable() for r in sm.getListOfRules()}
    for rxn in sm.getListOfReactions():
        net: dict[str, float] = {}
        for sign, refs in ((-1, rxn.getListOfReactants()), (1, rxn.getListOfProducts())):
            for ref in refs:
                net[ref.getSpecies()] = net.get(ref.getSpecies(), 0) + sign * ref.getStoichiometry()
        changed |= {sid for sid, n in net.items() if n != 0}
    constant = {s.getId() for s in sm.getListOfSpecies() if not s.getConstant()} - changed
    readers: dict[str, set[str]] = {}
    odf, mdf = problem.observable_df, problem.measurement_df
    if odf is not None and NOISE_FORMULA in odf.columns:
        for obs_id, formula in odf[NOISE_FORMULA].items():
            for sid in formula_symbols(formula) & constant:
                readers.setdefault(sid, set()).add(str(obs_id))
    if mdf is not None and NOISE_PARAMETERS in mdf.columns:
        for obs_id, cell in zip(mdf[OBSERVABLE_ID], mdf[NOISE_PARAMETERS]):
            for sid in formula_symbols(cell) & constant:
                readers.setdefault(sid, set()).add(str(obs_id))
    return [
        {
            "level": "error",
            "message": (
                f"the noise of {', '.join(f'`{o}`' for o in sorted(readers[sid]))} reads species "
                f"`{sid}`, which no reaction changes: AMICI holds such a species as a constant "
                "total and cannot build a noise model that reads one, so the fit would fail. Put "
                f"`{sid}` in a reaction that produces or consumes it."
            ),
            "check": "noise_reads_constant_species",
        }
        for sid in sorted(readers)
    ]


def _vanishes_at(node, zero: set[str], ruled: dict, seen: frozenset = frozenset()) -> bool:
    """Whether the math ``node`` is 0 wherever every name in ``zero`` is 0, read off its shape: such
    a name, or an assignment rule's target whose math vanishes (``ruled``); a literal 0; a product
    with a vanishing factor; a quotient with a vanishing numerator; a sum or difference of vanishing
    terms; a vanishing base to a positive literal power. Anything else is taken not to vanish, so a
    rate law this cannot read counts as one that runs and nothing is warned about."""

    def at(child) -> bool:
        return _vanishes_at(child, zero, ruled, seen)

    kind = node.getType()
    children = [node.getChild(i) for i in range(node.getNumChildren())]
    if kind == libsbml.AST_NAME:
        name = node.getName()
        return name in zero or (
            name in ruled
            and name not in seen
            and _vanishes_at(ruled[name], zero, ruled, seen | {name})
        )
    if node.isNumber():
        return node.getValue() == 0
    if kind == libsbml.AST_TIMES:
        return any(map(at, children))
    if kind == libsbml.AST_DIVIDE:
        return at(children[0])
    if kind in (libsbml.AST_PLUS, libsbml.AST_MINUS):
        return all(map(at, children))
    if kind in (libsbml.AST_POWER, libsbml.AST_FUNCTION_POWER):
        base, exponent = children
        return exponent.isNumber() and exponent.getValue() > 0 and at(base)
    return False


def _fixed_zero_parameters(problem) -> set[str]:
    """The parameters the parameter table fixes at 0 (a shipped ``B_init`` of an empty start)."""
    pdf = getattr(problem, "parameter_df", None)
    if pdf is None or not {ESTIMATE, NOMINAL_VALUE} <= set(pdf.columns):
        return set()
    out = set()
    for pid, estimate, nominal in zip(pdf.index, pdf[ESTIMATE], pdf[NOMINAL_VALUE]):
        try:
            if not as_bool(estimate) and _is_zero(nominal):
                out.add(str(pid))
        except ValueError:
            continue
    return out


def _is_zero(value) -> bool:
    """Whether a table cell is the number 0; an expression is taken not to be."""
    try:
        return float(value) == 0
    except (TypeError, ValueError):
        return False


# -- authoring-time check: frozen parametric reaction stoichiometry (checks the CONVERTED SBML) --
def frozen_parametric_stoichiometry_issues(sbml_model) -> list[dict[str, Any]]:
    """Hard-error when a reaction stoichiometry was written as a PARAMETER/symbol in Antimony.

    When an Antimony reaction uses a symbol as a stoichiometry coefficient — e.g.
    ``A -> k_amp B`` with ``k_amp`` a declared parameter — libantimony's SBML
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


def validate_problem(problem, protected_fixed_ids=None, statements: bool = False) -> dict[str, Any]:
    """Run PEtab v2 linting + libsbml SBML consistency and return a JSON-friendly report.

    Returns ``valid`` (no errors from either check), ``n_errors``/``n_warnings``, and the
    captured ``issues`` (level/message/check) — petab table issues first, then SBML ones.

    ``protected_fixed_ids`` (the shipped-fixed mechanistic/structural constants, from
    :meth:`session.PetabSession.protected_fixed_parameter_ids`) enables the role check that
    rejects removing or estimating one (see :func:`_fixed_constant_role_issues`); omit it to
    skip that check. ``statements`` words the readout check's remedy for make_changes'
    statements route.
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
    role_errors = [i for i in role_issues if i["level"] == "error"]
    n_errors += len(role_errors)
    n_warnings += len(role_issues) - len(role_errors)

    readout_issues = _undefined_readout_issues(problem, statements)
    issues += readout_issues
    n_errors += len(readout_issues)

    noise_issues = _constant_noise_species_issues(problem)
    issues += noise_issues
    n_errors += len(noise_issues)

    advisory = (
        _first_period_only_issues(problem)
        + _unread_input_issues(problem)
        + _sourceless_species_issues(problem)
    )
    issues += advisory
    n_warnings += len(advisory)

    return {
        "valid": (
            (not has_errors)
            and sbml_errors == 0
            and not reuse_errors
            and not role_errors
            and not readout_issues
            and not noise_issues
        ),
        "n_errors": n_errors,
        "n_warnings": n_warnings,
        "issues": issues,
    }
