"""Session state: a PEtab v2 problem with a fixed design, an Antimony-canonical model,
and persisted provenance/versioning.

The experimental design (measurements, conditions, experiments, observables) is fixed. The model
(edited only through the constrained operations in :mod:`operations`) and the parameter
table are mutable, and every committed change is snapshotted as a version and logged.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time
from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from petab.v2.C import CONDITION_ID, ESTIMATE, EXPERIMENT_ID, TARGET_ID
from petab.v2.models.sbml_model import SbmlModel

from . import criteria, logging_setup, protocols
from .constants import (
    ADEQUATE_GOF_PVALUE,
    DEFAULT_AUTO_FIT_STARTS,
    MAX_FIT_STARTS_FACTOR,
    NON_IMPROVING_STREAK_CAP,
    OVERFIT_GOF_PVALUE,
    PLATEAU_MIN_SCORE_GAIN,
    PLATEAU_WINDOW,
    STREAK_RESET_MIN_SCORE_GAIN,
)
from .data_access import residual_flag, residual_structure_rollup
from .errors import (
    FitCancelledError,
    InfrastructureError,
    ModelUpdateError,
    PetabMcpError,
    ValidationError,
    disk_full_error,
)
from .fitting import (
    FIT_PER_START_TIMEOUT_S,
    noise_sigma_collapsed,
    noise_sigma_inflated,
    table_fitted_values,
)
from .parameters import problem_fingerprint, sha16
from .problem import WorkingProblem, as_bool
from .provenance import ProvenanceStore
from .references import EutilsReferenceResolver

# Appended to an early-stop ``stop_reason`` only when the best model is NOT adequate (see
# progress_report). The nudge stays model-agnostic (issue #82): it names generic structural MOVE
# CATEGORIES, never a specific mechanism, so it can't hand over the answer to a problem whose model
# is withheld. It claims only what a streak shows. It once called the stall "not an optimizer or
# bounds problem", while dessauges 90 (gemot#134) ended on a streak made mostly of exactly
# equivalent rewrites of its best structure that failed to find its optimum again: an optimizer
# outcome, each. Either way, a refit or an equivalent rewrite cannot beat the optimum its structure
# already has.
_STRUCTURAL_PLATEAU_ADVICE = (
    "The best model is still not adequate, and none of these fits beat it. A refit or an "
    "equivalent rewrite of a structure already fitted can at best find that structure's optimum "
    "again (one that misses it scores worse), so neither can make the model adequate: that takes a "
    "STRUCTURAL change to the model's FORM, not another fit of the same structure. Consider a "
    "generic structural move (categories, not a specific answer) — add a latent / unobserved "
    "state; add a feedback / regulation reaction; or raise the interaction order (linear -> "
    "nonlinear / higher-order)."
)

# Appended instead of the structural nudge when a stall happened without ever scoring a model.
# Not a modelling problem: the likelihood could not be evaluated (typically the simulation does
# not reach every measured timepoint), so no fit produced a comparable score.
_UNSCORED_STALL_ADVICE = (
    "No fit in this session produced a usable score, so there is no best model to compare against: "
    "this is an EVALUATION failure, not evidence about the model's form. The likelihood is "
    "undefined when the simulation does not reach every measured timepoint — check "
    "n_missing_simulations and the per-point records for unsimulated points, "
    "and address why the trajectory stops (stiffness / a parameter pinned at a bound / an "
    "unbounded rate) before drawing any conclusion about the structure."
)

# Appended to the structural nudge when the best model's chi2 GoF only looks adequate because an
# ESTIMATED noise sigma inflated to absorb the misfit (see fitting.noise_sigma_inflated / constants.
# NOISE_SIGMA_MAX_FRACTION_OF_DATA_STD). The chi2 GoF test is then powerless — it rescales to
# whatever sigma makes chi2/dof ~= 1 — so an underfit model reads as "adequate" and the session
# would stop early on it. The fix is mechanistic, not noise-side.
_SIGMA_INFLATED_ADVICE = (
    " NOTE: the best model's fit only looks statistically adequate because its ESTIMATED noise "
    "sigma is inflated above its data-driven cap, absorbing the misfit — the chi2 goodness-of-fit "
    "is not real adequacy here. Do NOT widen or lean on the noise; fix the mechanism so the "
    "residuals shrink at a plausible sigma (the fit result's residual_structure verdict still "
    "flags the structure)."
)

# The other tail: the best model's ESTIMATED noise sigma collapsed implausibly small, so the fit
# is explaining nearly all the data's variability — overfitting the noise for a good-looking score.
_SIGMA_COLLAPSED_ADVICE = (
    " NOTE: the best model's ESTIMATED noise sigma has collapsed far below the data's own "
    "variability — the fit is absorbing nearly all the data into the mean, the fingerprint of "
    "overfitting the noise, so its low score is not genuine adequacy. Prefer a simpler / more "
    "parsimonious mechanism at a plausible sigma rather than a more flexible one that drives the "
    "residuals to ~0."
)

# A fit's optimum only 1 or 2 starts of a raised multistart reach (PEtab-MCP#287), for the
# `n_starts` cap's refusal and the warning on such a fit.
_NEEDLE_ADVICE = (
    "the optimum is a needle, and more starts will not make it reliable: reformulate the model "
    "where it is that sharp (a switch-like term, e.g. a steep Hill exponent or a gate at a bound; "
    "`get_fit_diagnosis(sections='parameters')`)"
)

#: The :meth:`PetabSession.progress_report` keys a fit result carries, each only when set;
#: ``get_progress`` has the rest. ``streak_cap`` and ``plateau_window`` are constants the
#: ``make_changes`` description states, ``criterion`` is on the fit block, and the best's score and
#: GoF p-value are on its own result.
_PROGRESS_ON_RESULT = (
    "n_fitted_ops",
    "non_improving_streak",
    "best_version",
    "score_improvement_over_last_n",
    "should_stop",
    "stop_reason",
)


def _n_unique(df, column: str) -> int:
    """How many distinct ``column`` ids ``df`` names; 0 for an absent table."""
    return 0 if df is None or column not in df.columns else int(df[column].nunique())


def _unique_ids(df, column: str) -> list[str]:
    """The distinct ``column`` ids ``df`` names, in its order; none for an absent table."""
    return [] if df is None or column not in df.columns else list(map(str, df[column].unique()))


def _with_fitted_values(simulation: dict[str, Any] | None, parameter_df) -> dict[str, Any] | None:
    """A version's stored ``simulation`` with the ``fitted_parameters`` its ``parameter_df``
    holds, where the simulation is a fit's: a version keeps them by name only in its table."""
    if simulation and simulation.get("status") in ("fit", "scored_at_nominal"):
        simulation["fitted_parameters"] = table_fitted_values(parameter_df)
    return simulation


@dataclass(frozen=True)
class Checkpoint:
    """The best-scoring fitted model so far — the version a client restores and scores, and the
    reference :func:`fitting.compute_improvement` gates the next fit against.

    One object rather than eight attributes because the eight are only ever meaningful together:
    a checkpoint is adopted or it is not. Adopting one used to be eight sequential assignments,
    and reconstructing one on a durable resume eight more — two places to keep in step whenever
    the adequacy gate grew a term.

    ``score`` is the session's complexity criterion (BIC by default — see :mod:`criteria`), and is
    finite by construction (see :meth:`PetabSession.record_fit_progress`): a NaN checkpoint can
    never be beaten, because every later ``score < NaN`` is False.
    """

    version: int
    score: float
    nll: float | None
    gof_pvalue: float | None
    sigma_estimated: bool
    sigma_inflated: bool
    sigma_collapsed: bool

    def __post_init__(self) -> None:
        if self.score is None or not math.isfinite(self.score):
            raise ValueError(
                f"a checkpoint's score must be a real number, got {self.score!r}: a non-finite "
                "one can never be beaten, because every later `score < NaN` is False."
            )

    @property
    def sigma_plausible(self) -> bool:
        """Whether the noise model is defensible — the sigma success gate in one place.

        True for a FIXED sigma (nothing was fitted, so nothing can be gamed), which is what lets
        a fixed-noise checkpoint resist being displaced by a better-scoring gamed one.
        """
        return not (self.sigma_inflated or self.sigma_collapsed)

    @property
    def sigma_gamed(self) -> bool:
        """Whether the score was bought by an ESTIMATED sigma outside the plausibility band."""
        return bool(self.sigma_estimated) and not self.sigma_plausible


def store_root(yaml_path: str | Path, provenance_dir: str | Path | None = None) -> Path:
    """A problem's provenance store: ``provenance_dir``, else ``<problem dir>/.petab_mcp``."""
    return (
        Path(provenance_dir) if provenance_dir else Path(yaml_path).resolve().parent / ".petab_mcp"
    )


class PetabSession:
    """Holds the working problem, provenance store, and reference resolver."""

    def __init__(
        self,
        yaml_path: str | Path,
        provenance_dir: str | Path | None = None,
        reference_mode: str = "best_effort",
        resolver: Any | None = None,
        criterion: "criteria.Criterion | str | None" = None,
    ) -> None:
        self.yaml_path = Path(yaml_path).resolve()
        if not self.yaml_path.is_file():
            raise FileNotFoundError(f"PEtab YAML not found: {self.yaml_path}")
        # The complexity criterion this session ranks models by (see :mod:`criteria`): the
        # explicit argument, else PETAB_MCP_CRITERION, else BIC — resolved once, here, and handed
        # to every scorer from this attribute. Set before anything that scores (the provenance
        # seed below reads it), and never reassigned by the session: a score is only comparable
        # against scores computed the same way.
        self.criterion: criteria.Criterion = criteria.resolve(criterion)
        # A v1 problem is upgraded here, once, and everything downstream sees v2 (see
        # petab_mcp.problem).
        self.problem: WorkingProblem = WorkingProblem.from_yaml(self.yaml_path)
        # The parameter table exactly as the problem SHIPPED it, captured before anything can
        # touch it — the reference for "which constants did the problem fix?" (see
        # `protected_fixed_parameter_ids`). It used to be read back from version 0, which was the
        # same thing only for as long as v0 was a verbatim snapshot of the load; the content-free
        # seed reconciles v0, and a durable resume replaces the working table wholesale. Reading
        # the YAML's own table is true on every process start, resume included, because
        # `WorkingProblem.from_yaml` above always re-reads the shipped `parameters.tsv`.
        self.shipped_parameter_df = self.problem.parameter_df.copy(deep=True)
        # What the experiments impose over time, from the `inputs.tsv` beside the YAML.
        self.inputs: tuple[protocols.Input, ...] = protocols.load(
            self.yaml_path.parent, self.problem
        )

        # Antimony is the canonical model form: prefer a model.ant next to the YAML.
        ant_file = self.yaml_path.parent / "model.ant"
        if ant_file.is_file():
            self._set_model_from_antimony(ant_file.read_text(encoding="utf-8"))
        elif self.problem.model is None:
            # No Antimony source and no SBML in the YAML: start from an empty model so the
            # working object always exists. `_seed_initial_model` below replaces it with a
            # content-free but VALID one; the empty skeleton survives only if seeding fails.
            from . import reconcile as R

            self._set_model_from_antimony(f"model {R.ZERO_SHOT_MODEL_ID}\nend")

        self.lock = threading.RLock()
        # Held only while the timing accumulators and the early-stop state below are updated
        # (`add_timing`, `record_fit_progress`, `reseed_best`) or read whole (`run_timing`,
        # `progress_report`); the reads take it instead of `lock`.
        self.counters_lock = threading.Lock()
        # When the change holding `lock` (a `make_changes` and its fit) began, by time.monotonic(),
        # else None; read without the lock, so a second change is refused instead of queued (see
        # `change`).
        self.change_began: float | None = None
        # The last change's starting point, the highest version stored when it began (None before
        # one has run), and its fit's progress (see `change`), which `wait_for_fit` reads.
        self.change_from: int | None = None
        self.change_progress: Any | None = None
        self.provenance = ProvenanceStore(store_root(self.yaml_path, provenance_dir))
        self.resolver = resolver or EutilsReferenceResolver(mode=reference_mode)
        self.logger = logging_setup.configure(self.provenance.root)
        # pyPESTO's own optimizer logs go to a separate <root>/pypesto.log
        logging_setup.configure_pypesto(self.provenance.root)
        # Rate-limit AMICI's per-non-finite-value warnings. Must happen BEFORE the first
        # `import amici` anywhere in the process — that is what lets us own the `amici` logger
        # instead of AMICI's own unfiltered handler (see `configure_amici`), and it is why this
        # call belongs here, in the first thing the server constructs, rather than next to the
        # fitting code that provokes the warnings.
        logging_setup.configure_amici()

        # Ids of parameters that ship FIXED and are structural/mechanistic constants — the ones
        # the agent must neither free nor remove. Derived once from the initial (v0) problem and
        # cached (the shipped design never changes); see `protected_fixed_parameter_ids`.
        self._protected_fixed_ids: set[str] | None = None
        # The model as first constructed (shipped or seeded), for classifying author-named
        # initial-condition parameters in `_compute_protected_fixed_ids`.
        self._initial_model_antimony: str | None = None

        # The engine is wired by the server after construction (it needs the session); until then,
        # and in engine-free unit tests, the fit hook no-ops. `simulation` is the current
        # per-point simulation dataframe.
        self.engine: Any | None = None
        self.simulation: dict[str, Any] | None = None
        # `(problem_hash, nll)` of the last version applied: the score it was committed with, and
        # the state that score belongs to (see `fitting._committed_score_note`).
        self.version_score: tuple[str, Any] | None = None
        # pyPESTO traces/results of the last fit (live diagnostic; not persisted per version).
        self.optimization: dict[str, Any] | None = None
        # The frozen record of the last fit — its built pyPESTO problem, its optimum, its tables
        # (see `fitting.FitRecord`). EVERY diagnosis of a fit reads this and nothing live, which is
        # what makes it impossible for a diagnosis to trigger an AMICI build.
        self.fit_record: Any | None = None
        self.auto_fit: bool = True
        self.auto_fit_starts: int = DEFAULT_AUTO_FIT_STARTS
        # Each start's wall-clock cap in seconds, for this session's fits and held-out refits; a
        # server hosting several problems sets it per session.
        self.fit_per_start_timeout: float = FIT_PER_START_TIMEOUT_S

        # Cumulative server-side timing for this session, split so the two very different costs
        # are legible: `compile_seconds` is AMICI model BUILD time (the one-time C++ compile
        # per unique structure — ONE build now, made by `fitting.estimation_problem` and shared
        # by the fit and every objective evaluation), `fit_seconds`
        # is pyPESTO OPTIMIZATION time (the recurring multistart `minimize`). Both accumulate
        # only when work actually happens (a compile/fit cache hit adds nothing), through
        # `add_timing`. Read via `run_timing` (`get_run_timing`).
        # `queue_seconds` is the part of `fit_seconds` its starts waited with none running.
        self.compile_seconds: float = 0.0
        self.fit_seconds: float = 0.0
        self.queue_seconds: float = 0.0
        self.n_compiles: int = 0
        self.n_fits: int = 0

        # Early-stop chain state (see constants.py). Every genuine fitted commit is recorded as
        # one "op" so a client can stop a session that has stopped improving and score the BEST
        # model rather than the last-committed one. ``is_new_best`` tracks the session's criterion
        # (see fitting.compute_improvement), so `best` IS the version at which the last new best
        # fired, the lowest score so far:
        #   * ``non_improving_streak`` — consecutive fitted ops since a new best last counted as
        #     progress; the hard stall cap fires on it.
        #   * ``streak_mark`` — the best score when the streak last reset; a new best resets it
        #     only ``STREAK_RESET_MIN_SCORE_GAIN`` below this.
        #   * ``best`` — the best-scoring fitted model so far (see :class:`Checkpoint`).
        #   * ``op_history`` — per-op (version, score, running best_score, gof p, is_new_best, and
        #     ``no_reset``: why a new best did not reset the streak), to surface "best-score
        #     improvement over the last N ops", to detect a plateau, and for the cap's stop_reason.
        # Every write holds ``counters_lock``, and ``progress_report`` reads them under it.
        self.non_improving_streak: int = 0
        self.streak_mark: float | None = None
        self.best: Checkpoint | None = None
        self.op_history: list[dict[str, Any]] = []

        # The session's OUTSTANDING invalid "does adding structure help?" experiment (see
        # :mod:`nesting` and `_check_nesting_vs_parent`): a commit that gained parameters yet fit
        # WORSE than its parent, which for a nested model is impossible — so the comparison cannot
        # support "the added structure does not help". Sticky across a rollback (the botched
        # variant is usually rolled back immediately, and HEAD's meta then carries nothing), so
        # `fit_summary` can hand it to a client's finish check. Cleared by a later commit that adds
        # parameters WITHOUT worsening the nll — a comparison that CAN answer the question.
        self.invalid_comparison: str | None = None

        # The Antimony the CALLER actually submitted for the next commit — the agent's own text,
        # before the reconcile adds declarations and before libantimony re-renders it. Stored
        # alongside the committed model as ``model_source.ant`` (see _snapshot_version).
        #
        # WHY IT IS WORTH A FILE. What every version stores today is the BUILD OUTPUT, and three
        # separate investigations broke on that:
        #   * LINEAGE cannot be recovered from it. A child's stored text is a re-render, so
        #     "which version was this patched from" is not answerable by text distance —
        #     measured, the attempt mis-assigned 43% of parents on sessions that could not even
        #     fork. A SOURCE changes only by the agent's own patches, so `child.source ==
        #     parent.source + patches` is checkable rather than inferable.
        #   * ANALYSIS of what the agent did (which patches landed, whether an edit was additive)
        #     has to reverse-engineer intent out of a rendered artifact.
        #   * The render/source gap is itself a measured failure mode: 19% of failed patch
        #     anchors matched the agent's own text but not the render.
        # Nothing READS this yet — it is provenance, not behaviour. The agent still sees the
        # committed model, because the reconcile's additions (measured: 19 symbols per session,
        # 3 of them minted `*_init` parameters it is actually fitting) are real and must not
        # vanish from what it reasons about.
        self.submitted_source: str | None = None

        # The CURRENT branch's model source: the text the caller last authored (`antimony=`) or
        # last edited (`model_edits`), carried across calls so a structured edit never has to
        # re-derive it from the render. Restored on rollback with the rest of the state.
        self.model_source: str | None = None
        # The statements record (:func:`operations.statements_to_antimony`) of the model last built
        # from statements, beside its source: what `get_model(format='statements')` renders, what
        # a commit that submits no model text inherits, and restored with the version.
        self.model_statements: dict[str, Any] | None = None

        # What the `statements` route asserted for the next commit: each row with the Emission
        # it produced (:func:`operations.statements_to_antimony`). Stored as ``statements.json``
        # beside the source it generated, and cleared with ``submitted_source``.
        self.submitted_statements: dict[str, Any] | None = None

        # The mechanisms the agent has WRITTEN this run, keyed (statement type, policy): the run's
        # additions to the library (:func:`mechanisms.written`), usable by every later call.
        # Each version's statements.json carries those the call that built its rows wrote.
        self.mechanisms: dict[tuple[str, str], dict[str, Any]] = {}

        # Species id -> INDRA Agent (:func:`seeding.load_entity_map`): what binds a model built
        # from statements to the fixed observable table. Empty unless the server was given one.
        self.entities: dict[str, Any] = {}

        # The claims every model must make (:mod:`constraints`), given as the server starts, or
        # None: checked on each change after its regulators are signed (operations._commit_change).
        self.constraints: Any | None = None
        # Whether `set_constraints` may change them: "off" (the tool is not offered) or "any".
        self.constraint_updates: str = "off"
        # Each stored version's verdict on a constraint, by (version, its text, the declared active
        # forms): tracing runs BioNetGen, and a version's model never changes.
        self._verdicts: dict[tuple, Any] = {}

        # The version the NEXT commit descends from, or None for "whatever HEAD is".
        # `make_changes(edits=..., base_version=N)` patches version N's stored text, so the commit
        # that follows is a child of N even when later versions exist. The server sets this for
        # the duration of such a call (see server.make_changes); everything else leaves it None
        # and the snapshot falls back to HEAD, which is correct for an edit of the working model.
        self.derived_from: int | None = None
        # The `(old, new)` of each `rename` in the running `make_changes` call's `model_edits`
        # (:func:`model_edits.renames`), set by the server for the call like `derived_from`: the
        # reconcile carries each old id's row to its new id (`operations.set_model`), and the fit
        # its warm-start coordinate (`fitting._run_fit`).
        self.renamed: list[tuple[str, str]] = []

        # Which `make_changes` routes the server offers: True for the text routes, whose
        # `model_edits` take a merge hint as a `rename` (`identifiability.merge_hints`).
        self.text_route: bool = True

        # De-duplication memory for identical fits (see :mod:`dedup` and issue #59): how many times
        # each *identical* estimation problem has been handed to `fit` this session, keyed by
        # structure fingerprint + estimation setup + starts (see `dedup.reauthor_key`) so that only
        # truly identical problems — not the same structure with changed bounds/scale or more
        # starts — count toward the loop guard. A within-session counter (a fresh session
        # legitimately starts over), not persisted.
        self._fit_attempts: dict[str, int] = {}
        # The edit and lint warnings the last committed `make_changes` result carried, each with
        # the version whose result first carried it in full (see `operations.bundle_result`).
        self.warnings_sent: dict[tuple[str, str], int] = {}

        if self.provenance.current_version < 0:
            # The initial model is WORKING STATE, not a version: every version is a fit, and
            # nothing has been fit yet. The empty skeleton is the one state nothing downstream can
            # survive, so it is replaced with a valid one here (see `_seed_initial_model`).
            self._seed_initial_model()
            self._initial_model_antimony = self.model.to_antimony()
        else:
            # Capture the model the session STARTS from BEFORE `apply_version` swaps in HEAD's:
            # `_compute_protected_fixed_ids` classifies initial-condition parameters against it,
            # and HEAD's model declares a different set. A model-less problem still holds only the
            # empty skeleton here, so regenerate the seed it would have been given — pure with
            # respect to the fixed tables, which never change.
            from . import reconcile as R

            self._initial_model_antimony = (
                self.model.to_antimony()
                if self.has_model()
                else R.seed_model_antimony(self.problem)
            )
            # durable resume: restore the working state from the latest version
            self.apply_version(self.provenance.current_version)
            # ...and the mechanisms the run wrote, off every version's record, since a row naming
            # one must still build and none may be redefined: those its call wrote, which a row
            # need not use, and those its rows used (a record may predate `mechanisms`). Oldest
            # first, so the first definition of a name wins, as it does live.
            for meta in self.provenance.list_versions():
                record = self.provenance.get_statements(meta["version"]) or {}
                for d in record.get("mechanisms", []) + [
                    row["mechanism"] for row in record.get("rows", []) if "mechanism" in row
                ]:
                    self.mechanisms.setdefault((d["statement_type"], d["policy"]), d)
            # apply_version restores `model_source` too, so a durable resume keeps editing the
            # SOURCE rather than silently adopting the render as its base — which would break
            # `child.source == parent.source + ops` exactly at the resume boundary, with nothing
            # in the payload marking where.
            # ...and seed the best-score checkpoint from the prior events, so the first commit after
            # a resume is judged against the session's real best rather than treated as a free new
            # best.
            self._seed_progress_from_provenance()

    def _seed_initial_model(self) -> None:
        """Replace a model-less problem's empty skeleton with a CONTENT-FREE but VALID model.

        The empty skeleton is not a neutral "no model yet" placeholder: it is a broken PEtab
        problem whose observables reference entities nothing defines, and AMICI's SBML import
        SEGFAULTS on it — killing the server process, which no caller can catch or degrade. Every
        path that touches AMICI therefore had to be guarded against reaching it, and v0 could not
        be restored at all.

        Seeding removes the state instead of guarding it. :func:`reconcile.seed_model_antimony`
        declares one species per identifier the FIXED tables already name and nothing else — no
        reactions, no rules — so v0 becomes importable, simulatable and restorable while carrying
        no mechanism (see that function for why this leaks nothing). ``model_authored`` stays
        False until a ``make_changes`` call adds actual dynamics, so the model-less contract is
        unchanged.

        The seed is RECONCILED like any other model, except that its species get no parametric
        initial condition (``parametric_ics=False``). It declares no mechanism and no
        ``<sid>_init``, so the kinetic and initial-condition rows a model-less problem ships are
        pruned at v0; that is safe because :func:`reconcile.sync_parameter_table` RESTORES a
        shipped row the moment the caller's model reintroduces that parameter, its bounds intact.

        It then LINTS, under the same rules every edit is held to, and a failure RAISES. It
        deliberately does not degrade to the empty skeleton: that state is the one nothing
        downstream survives (AMICI's SBML import segfaults on it, killing the process), and
        leaving it reachable is what forced a ``has_model()`` check into every path that touches
        AMICI. Refusing to construct instead means a problem whose fixed tables cannot produce a
        valid model fails at startup, naming the lint errors, rather than starting and dying
        later with no traceback — and it is what makes "there is always a model" a real
        invariant rather than an expectation.
        """
        if self.has_model():
            return
        from . import reconcile as R
        from .validation import validate_problem

        antimony = R.seed_model_antimony(self.problem)
        if antimony is None:
            raise ValidationError(
                "this problem ships without a model, and its fixed tables name nothing that "
                "could be declared as one: no observable formula references an entity, no "
                "condition targets one, and the parameter table is empty. There is "
                "nothing to simulate and nothing an initial model could be built from."
            )
        try:
            self._set_model_from_antimony(antimony)
            R.reconcile(self.problem, shipped=self.shipped_parameter_df, parametric_ics=False)
        except Exception as exc:
            raise ValidationError(
                f"could not build a valid initial model for this model-less problem "
                f"({type(exc).__name__}: {exc}). The seed is generated from the fixed "
                "observable/condition/parameter tables, so this means those tables name "
                "something that cannot be declared as a model entity."
            ) from exc
        # The invariant is ENFORCED here, not assumed. The seed can lint perfectly and still
        # declare no SPECIES — a problem whose observable formulas are all constants names no
        # entity, so only the parameter rows get declared — which lands right back in the state
        # this method exists to remove. Nothing downstream would notice until AMICI segfaulted
        # on it, so check the predicate itself rather than trusting the construction.
        if not self.has_model():
            raise ValidationError(
                "this problem ships without a model and none can be built from its fixed tables: "
                "no observable formula references a model entity and no condition targets one, "
                "so there are no species to simulate. Nothing here can be fit."
            )
        report = validate_problem(self.problem)
        if not report["valid"]:
            raise ValidationError(
                "the initial model generated for this model-less problem does not validate, so "
                "the problem cannot be worked on: its fixed tables and the model they imply are "
                "inconsistent.",
                issues=[
                    i["message"] for i in report["issues"] if i["level"] in ("error", "critical")
                ],
            )
        self.logger.info(
            "seeded a content-free initial model: %d species, no reactions",
            self.model.sbml_model.getNumSpecies(),
        )

    def _seed_progress_from_provenance(self, excluded: frozenset[int] = frozenset()) -> None:
        """Seed the best-score :class:`Checkpoint` from prior events (durable resume).

        Replays the fit events in the log, scored under the session's criterion (see
        :meth:`criteria.Criterion.score_of`), through the live gates —
        :func:`fitting.beats_best` and :meth:`_adopt_checkpoint` — so the checkpoint is
        the one the last process kept, not the lowest score it saw: that may be a sigma-gamed fit
        the live run refused. ``compute_improvement`` and the streak then pick up where the last
        process left off. The streak and ``op_history`` stay reset — they
        are within-session heuristics whose exact sequence isn't reconstructed — and
        ``streak_mark`` is the seeded best's score, as if the streak had just reset. Best-effort: an
        event that carries no score under THIS criterion simply doesn't seed, which is also what
        makes resuming under a criterion the earlier process did not use safe — it starts fresh
        rather than ranking one criterion's scores against another's. Called under construction,
        so no lock is needed, and by :meth:`reseed_best`. The versions ``excluded`` (those the
        session's constraints rule out) seed nothing.

        Also restores ``invalid_comparison`` from the LAST fit event that recorded it — the
        session's outstanding invalid does-adding-structure-help experiment, which a finish check
        reads. It is deliberately not derived from the per-commit ``nesting_advisory`` fields: a
        later valid additive experiment clears the flag without writing an advisory of its own,
        so replaying advisories alone would resurrect one already superseded.
        """
        from . import fitting

        try:
            for ev in self.provenance.recent_events(0):
                if "invalid_comparison" in ev:
                    self.invalid_comparison = ev.get("invalid_comparison")
                # `score_of` already rejects non-finite values, so a resume can't seed the
                # checkpoint with a NaN that no later fit could beat (see record_fit_progress).
                score = self.criterion.score_of(ev)
                if score is None or ev.get("version") in excluded:
                    continue
                nll = ev.get("neg_log_likelihood")
                gofp = ev.get("chi2_gof_pvalue")
                # The adequacy verdict is persisted alongside the score, so a resumed best
                # keeps it. An older event without these fields reads False (fixed / not
                # inflated), which falls back to the GoF branch of the gate.
                self._adopt_checkpoint(
                    Checkpoint(
                        version=ev.get("version"),
                        score=score,
                        nll=float(nll) if nll is not None else None,
                        gof_pvalue=float(gofp) if gofp is not None else None,
                        sigma_estimated=bool(ev.get("noise_sigma_estimated", False)),
                        sigma_inflated=bool(ev.get("noise_sigma_inflated", False)),
                        sigma_collapsed=bool(ev.get("noise_sigma_collapsed", False)),
                    ),
                    fitting.beats_best(score, self.best.score if self.best else None),
                )
        except Exception:  # noqa: BLE001 - seeding is best-effort; a resume must never fail on it
            self.logger.warning(
                "could not seed best-score checkpoint from provenance", exc_info=True
            )
        self.streak_mark = self.best.score if self.best else None

    # -- constraints -------------------------------------------------------
    def constraint_verdicts(self, version: int | None, constraints: Any) -> dict[str, Any]:
        """Each of ``constraints``' verdicts on stored ``version`` (``None`` or none committed yet:
        the current model): as the version's record keeps it from its commit, as found before, or
        checked now, tracing the version's model (:func:`constraints.check`)."""
        from . import constraints as constraints_mod
        from .operations import statements_model

        stored = version is not None and version >= 0
        record = self.provenance.get_statements(version) if stored else self.model_statements

        def model():
            return statements_model(self, record)

        if not stored:
            return constraints_mod.check(constraints, record, model)
        forms = tuple(f.text for f in constraints.active_forms)
        kept = constraints_mod.kept(constraints, record)
        out, todo = {}, []
        for c in constraints.items:
            key = (version, c.text, forms)
            if key not in self._verdicts and c.id in kept:
                self._verdicts[key] = kept[c.id]
            if key in self._verdicts:
                out[c.id] = self._verdicts[key]
            else:
                todo.append(c)
        if todo:
            found = constraints_mod.check(
                constraints_mod.Constraints(tuple(todo), constraints.active_forms), record, model
            )
            for c in todo:
                out[c.id] = self._verdicts[(version, c.text, forms)] = found[c.id]
        return out

    def constraint_violations(self, constraints: Any | None = None) -> dict[int, list[str]]:
        """The ids of the constraints (the session's, or ``constraints``) each stored version
        breaks, by version (:meth:`constraint_verdicts`); a version breaking none is left out, and
        all are without constraints."""
        constraints = constraints if constraints is not None else self.constraints
        if constraints is None:
            return {}
        out = {}
        for meta in self.provenance.list_versions():
            if meta.get("version") is None:
                continue
            verdicts = self.constraint_verdicts(int(meta["version"]), constraints)
            if broken := [c.id for c in constraints.items if not verdicts[c.id].ok]:
                out[int(meta["version"])] = broken
        return out

    def reseed_best(self, excluded: set[int]) -> None:
        """Rebuild the best-score checkpoint from the log without the versions ``excluded``, the ones
        the constraints in force break: the best version, the streak's reset mark and the plateau
        stop then rank only models that make the session's claims."""
        with self.lock, self.counters_lock:
            self.best = None
            self._seed_progress_from_provenance(frozenset(excluded))

    def add_timing(self, **amounts: float) -> None:
        """Add each of ``amounts`` to the timing accumulator of its name (``fit_seconds=2.5,
        n_fits=1``), all under :attr:`counters_lock`, so :meth:`run_timing` sees all or none."""
        with self.counters_lock:
            for name, amount in amounts.items():
                setattr(self, name, getattr(self, name) + amount)

    def run_timing(self) -> dict[str, Any]:
        """The timing accumulators ``get_run_timing`` returns, read under :attr:`counters_lock`
        as :meth:`progress_report` reads the early-stop state, and never under :attr:`lock`: a
        change holds that for its whole edit and fit, so a read during a fit waited it out (29.6 s
        of a 30 s stub, PEtab-MCP#180) instead of answering with what the change has recorded."""
        with self.counters_lock:
            return {
                "compile_seconds": round(float(self.compile_seconds), 3),
                "fit_seconds": round(float(self.fit_seconds), 3),
                "queue_seconds": round(float(self.queue_seconds), 3),
                "n_compiles": int(self.n_compiles),
                "n_fits": int(self.n_fits),
            }

    def set_constraints(self, constraints: Any, rationale: str) -> dict[str, Any]:
        """Replace the session's constraints with ``constraints`` (:mod:`constraints`): the next change
        is checked against them, the versions they break are flagged and no longer restorable or
        ranked, and the update is logged with ``rationale`` and whether it loosens or tightens."""
        from . import constraints as constraints_mod

        with self.lock:
            old = self.constraints
            broken = self.constraint_violations(constraints)
            how = constraints_mod.change(old, constraints)
            head = self.provenance.current_version
            self.constraints = constraints
            self.reseed_best(set(broken))
            if self.model_statements:
                # What `list_constraints` reads: HEAD under the new set.
                verdicts = self.constraint_verdicts(head, constraints)
                self.model_statements["constraints"] = {
                    "checked": constraints.texts(),
                    "verdicts": {cid: v.stored() for cid, v in verdicts.items()},
                }
            self.provenance.append_event(
                {
                    "type": "constraints",
                    "constraints": constraints.texts(),
                    "previous": old.texts() if old is not None else [],
                    "change": how,
                    "rationale": rationale,
                    "head": head,
                    "violating": {str(v): ids for v, ids in broken.items()},
                }
            )
        return {
            "applied": True,
            "change": how,
            "constraints": constraints.texts(),
            "head_violates": broken.get(head, []) if head is not None else [],
            "violating_versions": {str(v): ids for v, ids in broken.items()},
            "best_version": self.best.version if self.best else None,
        }

    # -- model access ------------------------------------------------------
    @property
    def model(self):
        if self.problem.model is None:
            raise ValueError("The PEtab problem has no model.")
        return self.problem.model

    def has_model(self) -> bool:
        """Whether there is a model at all (any species or reactions).

        THIS IS NOT THE MODEL-LESS FLAG — use :meth:`model_authored` for that. A model-less
        problem is SEEDED at construction with a content-free but valid model (see
        :meth:`_seed_initial_model`), so this is normally True from the first moment, including
        for a model-less problem that has authored nothing.

        In fact this is now an INVARIANT rather than a question: it is True for the whole life of
        every session. A problem that ships a model has one; a problem that does not is seeded
        with one, and :meth:`_seed_initial_model` REFUSES TO CONSTRUCT the session if it cannot
        be (including the case where the seed lints but declares no species). ``make_changes``
        then refuses to commit a model without dynamics, so nothing can take it away again.

        That is what removed the ``has_model()`` checks that used to sit in ``fit``,
        ``ensure_simulation``, ``_restore_or_simulate`` and ``identifiability``. They existed for
        one state — the empty skeleton — which AMICI's SBML import segfaults on, taking the
        server process with it where no caller can catch or degrade. Making the state
        unreachable is what let every one of them go; the two call sites left are the seeding
        path itself, which establishes the invariant rather than defending against it.
        """
        model = self.problem.model
        if model is None:
            return False
        sm = model.sbml_model
        return sm.getNumReactions() > 0 or sm.getNumSpecies() > 0

    def model_authored(self) -> bool:
        """Whether a MECHANISM has been authored — the model-less flag, reported as
        ``model_authored`` by ``describe_problem``.

        True once the model has dynamics: at least one reaction, or a rate/assignment rule. The
        content-free seed (:meth:`_seed_initial_model`) has neither by construction, so it reads
        False exactly as the old empty skeleton did, and ``make_changes`` refuses to commit a
        model that would leave it False (nothing to simulate is nothing to score). So this flips
        exactly once, on the first successful authoring call, which is what a client's no-model
        checks key on.
        """
        from . import reconcile as R

        model = self.problem.model
        return model is not None and R.has_dynamics(model.sbml_model)

    def model_sbml(self) -> str:
        return self.model.to_sbml_str()

    def sbml_hash(self) -> str:
        return sha16(self.model_sbml())

    def module_hash(self) -> str:
        """The key of the compiled AMICI module this problem needs — the model plus the parts of
        the design and the estimated set the v2 import compiles in (see
        :meth:`problem.WorkingProblem.module_fingerprint`). The design is fixed for a session, so
        within one session this moves with the SBML and the estimated set."""
        return self.problem.module_fingerprint()

    def problem_hash(self) -> str:
        """Hash of the full problem identity: model SBML + the parameter table.

        Unlike ``module_hash`` (what the AMICI compile cache is keyed on), this changes when a
        parameter's bounds/scale/estimate/value change, so it uniquely
        identifies a *fit* and is the correct key for fit artifacts (the pyPESTO result
        cache). Keying those on the model alone would collide across versions that
        differ only in the parameter table.
        """
        return problem_fingerprint(self.model_sbml(), self.problem.parameter_df)

    def _set_model_from_antimony(self, antimony: str) -> None:
        rel = getattr(self.problem.model, "rel_path", None) if self.problem.model else None
        model = SbmlModel.from_antimony(
            antimony, model_id=getattr(self.problem.model, "model_id", None)
        )
        model.rel_path = rel or "model.xml"
        self.problem.model = model

    def protected_fixed_parameter_ids(self) -> set[str]:
        """Ids of parameters that ship FIXED (``estimate=0``) and are mechanistic/structural
        constants — a conservation total, a known rate, a measured concentration. These must not
        be freed *or* removed: validation rejects a commit that turns one estimated or drops it
        from the model (see ``validation._fixed_constant_role_issues``).

        Derived from the SHIPPED parameter table (``shipped_parameter_df``, captured at load)
        and cached — the shipped design never changes.
        EXCLUDED (each has its own handling, so neither is "protected"):

        * initial-condition parameters — removed legitimately when their species is removed;
          detected from the INITIAL model, with an ``*_init`` name fallback for the parameters a
          content-free seed cannot classify;
        * observation-model factors (noise sigmas, observable scaling/offset) — pinned by the
          fixed observable/measurement tables and given their own estimation paths.
        """
        if self._protected_fixed_ids is not None:
            return self._protected_fixed_ids
        self._protected_fixed_ids = self._compute_protected_fixed_ids()
        return self._protected_fixed_ids

    def _compute_protected_fixed_ids(self) -> set[str]:
        from . import reconcile as R

        ref_df = self.shipped_parameter_df
        if ref_df is None or ESTIMATE not in getattr(ref_df, "columns", []):
            return set()

        def _is_fixed(value: Any) -> bool:
            try:
                return not as_bool(value)
            except (TypeError, ValueError):
                return False

        fixed = {str(pid) for pid in ref_df.index if _is_fixed(ref_df.loc[pid, ESTIMATE])}
        if not fixed:
            return set()

        ic: set[str] = {p for p in fixed if p.endswith("_init")}
        if self._initial_model_antimony is not None:
            try:
                initial = SbmlModel.from_antimony(self._initial_model_antimony, model_id=None)
                ic |= {str(x) for x in R._ic_parameter_ids(initial.sbml_model)}
            except Exception:  # noqa: BLE001 - the *_init name fallback above still applies, and
                # partial protection beats none: raising here reaches _commit_change's catch, which
                # passes protected_fixed=None and skips the shipped-constant check altogether.
                self.logger.warning(
                    "could not read the initial model's initial-condition parameters; "
                    "falling back to the *_init name heuristic",
                    exc_info=True,
                )

        obs = {
            str(x)
            for x in (R.noise_sigma_ids(self.problem) | R.observable_scaling_ids(self.problem))
        }
        return fixed - ic - obs

    def describe(self) -> dict[str, Any]:
        """A summary of the problem — model size, tables, free parameters, current fit metrics.

        Backs the ``describe_problem`` tool. ``neg_log_likelihood``/``chi2`` come from the
        stored simulation and are ``None`` until the first commit/simulation.
        """
        from . import reconcile as R

        p = self.problem
        sm = self.model.sbml_model
        # How many claims every model must make, where the session has any; `list_constraints`
        # lists them, with what makes each hold (:mod:`constraints`).
        constrained = (
            {"n_constraints": len(self.constraints.items)} if self.constraints is not None else {}
        )
        authored = self.model_authored()
        # Counted on an authored model only, never on the seed's placeholders (docs/RATIONALE.md).
        counts = (
            {"n_species": int(sm.getNumSpecies()), "n_model_parameters": int(sm.getNumParameters())}
            if authored
            else {}
        )
        return constrained | {
            "model_id": self.model.model_id,
            # Has a MECHANISM been authored? False for a problem that ships without one, whose
            # initial version is a content-free seed (species only, no dynamics) rather than the
            # broken empty skeleton it used to be. Replaces `model_initialized`, which since the
            # seed is True from construction and no longer answers the question anyone asked.
            "model_authored": authored,
            **counts,
            "n_reactions": int(sm.getNumReactions()),
            "n_table_parameters": int(len(p.parameter_df)),
            "n_estimated": int(p.n_estimated),
            "free_parameters": [str(x) for x in p.x_free_ids],
            "readouts_to_define": [r for r, rule in R.readouts(p).items() if rule is None],
            "observables": [str(x) for x in getattr(p.observable_df, "index", [])],
            "n_conditions": _n_unique(p.condition_df, CONDITION_ID),
            "condition_targets": _unique_ids(p.condition_df, TARGET_ID),
            "n_experiments": _n_unique(p.experiment_df, EXPERIMENT_ID),
            "n_measurements": 0 if p.measurement_df is None else int(len(p.measurement_df)),
            "n_reactions_detail": [sm.getReaction(i).getId() for i in range(sm.getNumReactions())],
            "current_version": self.provenance.current_version,
            # The multistart count a fit uses when `n_starts` is omitted. Reported because
            # `n_starts` OVERRIDES it rather than adding to it: without knowing the number, a
            # caller cannot tell whether the value it passes fits harder or (silently) weaker,
            # and guessing low degrades every fit it makes.
            "fit_n_starts_default": int(self.auto_fit_starts),
            "fixed_tables": ["measurements", "conditions", "experiments", "observables"],
            "inputs": [{"input": i.id, "rule": i.rule} for i in self.inputs],
            "neg_log_likelihood": (self.simulation or {}).get("neg_log_likelihood"),
            "chi2": (self.simulation or {}).get("chi2"),
        }

    # -- versioning --------------------------------------------------------
    def apply_version(self, version: int) -> None:
        """Set the working model + parameter table to a stored version (fixed tables kept).

        The parameter table is restored directly as the stored DataFrame (no JSON records
        round-trip). Also restores the version's stored simulation dataframe (``None`` for
        legacy versions that predate it), with the fitted values its table holds. Stays
        engine-free: this runs during durable resume in ``__init__`` before the engine is wired.
        """
        with self.lock:
            snap = self.provenance.get_version(version)
            self._set_model_from_antimony(snap["model_antimony"])
            self.problem.parameter_df = self.provenance.get_parameter_df(version)
            self.simulation = _with_fitted_values(snap.get("simulation"), self.problem.parameter_df)
            self.version_score = (self.problem_hash(), snap.get("neg_log_likelihood"))
            # The branch source moves WITH the model. Without this, a structured edit after a
            # rollback applies its operations to the abandoned branch's text and silently
            # resurrects it. Falls back to the restored render for versions stored before
            # sources existed, which is the correct base for them.
            self.model_source = self.provenance.get_model_source(version) or snap["model_antimony"]
            # ...and so do its rows.
            self.model_statements = self.provenance.get_statements(version)
            # Restore this version's fit artifacts too: the optimization traces/results are
            # persisted per version (``None`` for legacy/unfitted versions), so a rollback or
            # durable resume brings them back rather than dropping them.
            self.optimization = snap.get("optimization")

    def stored_fit(self, version: int) -> SimpleNamespace:
        """Stored ``version``'s fit as :meth:`apply_version` would restore it, without touching the
        working state: ``problem`` is the working problem with that version's parameter table,
        ``simulation`` its stored records with the fitted values that table holds, ``optimization``
        ``None``. ``get_fit_diagnosis(version=N)`` hands it to the residual reads of
        :mod:`data_access` in place of the session. Those read the tables and the records, never the
        model, so ``problem`` keeps the working model and nothing is built or refit. Raises
        ``UnknownEntityError`` for a version that does not exist."""
        parameter_df = self.provenance.get_parameter_df(version)
        return SimpleNamespace(
            problem=self.problem.replace(parameter_df=parameter_df),
            simulation=_with_fitted_values(
                self.provenance.get_version_simulation(version), parameter_df
            ),
            optimization=None,
        )

    def _snapshot_version(
        self,
        *,
        meta: dict[str, Any],
        sim: dict[str, Any] | None,
        model_ant: str | None = None,
    ) -> int:
        """Snapshot the current model + parameter table as a new version.

        Every version is a fit, so there is one kind. Attaches the per-version simulation and
        optimization payloads; callers supply only ``meta`` and the simulation blob. ``model_ant``
        defaults to the current model's Antimony; pass it when the caller already serialized it.
        """
        from . import fitting
        from .parameters import structure_fingerprint

        # Stamp the structure fingerprint + the criterion scores into every version's meta so
        # the de-dup registry (see :func:`dedup.build_registry`) can recognise a re-authored
        # structure and rank structure families by fit. Computed once here so every commit path
        # records it uniformly; best-effort, since a fingerprint/score failure must never fail a
        # snapshot.
        meta = dict(meta or {})
        if "structure_hash" not in meta:
            try:
                meta["structure_hash"] = structure_fingerprint(
                    self.model.sbml_model, self.problem.parameter_df
                )
            except Exception:  # noqa: BLE001 - advisory metadata only
                pass
        nll = (sim or {}).get("neg_log_likelihood")
        fitted = (sim or {}).get("fitted_parameters")
        # k = complexity_k (n_estimated + hardcoded constants) — the honest complexity count. The
        # live problem/model is authoritative for the state being snapshotted when the payload
        # lacks the counts; len(fitted_parameters) can differ (it omits analytically-profiled
        # inner params), so it is only a last-resort fallback.
        k = fitting.complexity_k(sim)
        if k is None:
            try:
                k = fitting.complexity_k(
                    {
                        "n_estimated": int(self.problem.n_estimated),
                        "hardcoded_constants": self.hardcoded_constants(),
                    }
                )
            except Exception:  # noqa: BLE001 - scores are advisory metadata; never fail a snapshot
                k = len(fitted) if fitted is not None else None
        # EVERY criterion by name, not just the one this session ranks by: the meta is what the
        # de-dup registry ranks on and what a client reads back, and storing all of them means a
        # finished session can be re-ranked under a criterion it was not run with. Fill only what
        # the caller left open — a fit path already merged `compute_improvement`, whose scores
        # came off the same numbers, so this must never overwrite them with a k derived otherwise.
        scores = fitting.goodness_of_fit(
            nll,
            (sim or {}).get("chi2"),
            k,
            (sim or {}).get("n_data_points"),
            criterion=self.criterion,
            k_eff=(sim or {}).get("k_eff"),
        )
        for key in (*criteria.PENALTIES, "criterion", "criterion_value"):
            if meta.get(key) is None:
                meta[key] = scores[key]
        meta.setdefault("k", k)

        return self.provenance.snapshot_version(
            model_ant=self.model.to_antimony() if model_ant is None else model_ant,
            parameters=self.problem.parameter_df,
            label="fit",
            kind="fit",
            # LINEAGE, not chronology. `parent` defaults to HEAD, which is right for an edit of
            # the working model and WRONG for a fork: `make_changes(edits=..., base_version=N)`
            # patches version N's stored text, so the child descends from N however many other
            # versions were committed since. Measured on 30 diff-edit sessions: 207 of 311
            # base_version calls pointed at something older than HEAD, and every one of those
            # was recorded as a child of HEAD — 30/30 version trees looked perfectly linear
            # while two thirds of the forks were real. Everything that reasons parent-to-child
            # then compares against a model the child never came from: `_check_nesting_vs_parent`
            # (so the advisory describes a comparison that did not happen),
            # `fitting._slim_fitted_parameters` (diffs against the wrong predecessor), and any
            # later trajectory analysis.
            parent=self.derived_from,
            # A commit that changed only BOUNDS still has a source — the one it inherited. Writing
            # None there left the version with no `model_source.ant`, and the next resume or
            # rollback onto it fell back to the reconciled RENDER as the branch base: operations
            # then edit render text, `child.source == parent.source + ops` quietly stops holding,
            # and nothing in the payload marks where it stopped. Carry the current source forward
            # so every version after the first authoring has one.
            model_source=self.submitted_source or self.model_source,
            # The same for the rows: a commit that submits no model text (bounds only, a refit)
            # keeps the model its rows built, so it keeps the rows. One that submits text without
            # rows (`antimony`, `model_edits`) changed the model under them and records none.
            statements=(
                self.submitted_statements if self.submitted_source else self.model_statements
            ),
            meta=meta,
            simulation=fitting.build_sim_payload_for_version(sim),
            optimization=fitting.build_optimization_payload_for_version(self.optimization),
        )

    @property
    def max_fit_starts(self) -> int:
        """The most starts one fit may run: :data:`MAX_FIT_STARTS_FACTOR` times the default."""
        return MAX_FIT_STARTS_FACTOR * int(self.auto_fit_starts)

    def _coerce_n_starts(self, n_starts: int | None) -> int | None:
        """Validate ``n_starts`` — ``None`` (server default) or an integer from 1 to
        :attr:`max_fit_starts` — returning the coerced int. Raises :class:`ValidationError`
        otherwise.

        0 used to mean "simulate without optimising". It is refused now: every change is scored,
        and a version whose parameters were never optimised is not a score — it records whatever
        the nominal values happened to be, then competes on score against versions that were fit.
        The simulate-only pass itself is unaffected; it is still how a restore re-simulates a
        stored fit (``fit_and_simulate(do_fit=False)``), which is not a caller-reachable path.
        """
        if n_starts is None:
            return None
        try:
            n_starts = int(n_starts)
        except (TypeError, ValueError):
            raise ValidationError(f"n_starts must be an integer >= 1, got {n_starts!r}.")
        if n_starts < 1:
            raise ValidationError(f"n_starts must be an integer >= 1, got {n_starts}.")
        if n_starts > (cap := self.max_fit_starts):
            raise ValidationError(
                f"n_starts must be at most {cap}, {MAX_FIT_STARTS_FACTOR}x the default, got "
                f"{n_starts}. If a raise still leaves 1 or 2 starts at the best, {_NEEDLE_ADVICE}."
            )
        return n_starts

    # -- fit + simulate ----------------------------------------------------
    def _auto_fit_and_improvement(
        self, n_starts: int | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Fit + simulate for a ``fit`` call, returning (payload, improvement).

        ``n_starts`` is a multistart count >= 1; ``None`` uses the server default. No-ops (status
        ``"skipped"``) when no engine is wired or ``auto_fit`` is off. A fit/simulation error
        propagates to the caller.
        """
        from . import fitting

        n_starts = self._coerce_n_starts(n_starts)

        engine = self.engine
        if engine is None or not self.auto_fit:
            return {
                "status": "skipped",
                "neg_log_likelihood": None,
                "chi2": None,
                "fitted_parameters": None,
                "records": [],
            }, {}
        starts = self.auto_fit_starts if n_starts is None else n_starts
        # A fit needs at least one free parameter. Catch the zero-estimated case here with a clear,
        # actionable message rather than letting pyPESTO fail cryptically ("'NoneType' object is
        # not iterable") deep in the optimizer.
        if int(self.problem.n_estimated or 0) == 0:
            raise ModelUpdateError(
                "the model has NO estimated (free) parameters, so there is nothing to "
                "fit. Give it at least one free parameter — a new symbol you use in a "
                "rate law becomes an estimated parameter automatically, so declare one "
                "(or keep using the shipped free parameters) — then re-author with "
                "make_changes."
            )
        fitsim = fitting.fit_and_simulate(self, engine, do_fit=True, n_starts=starts)
        # The multistart count this fit ACTUALLY used, so a caller never has to infer whether its
        # `n_starts` raised or lowered the server default (`n_starts` overrides it).
        fitsim["n_starts"] = starts
        nllh = fitsim.get("neg_log_likelihood")
        improvement = (
            fitting.compute_improvement(self, nllh, fitsim.get("objective_before"), fitsim=fitsim)
            if nllh is not None
            else {}
        )
        return fitsim, improvement

    def _restore_or_simulate(self) -> dict[str, Any] | None:
        """Ensure ``self.simulation`` is populated after restoring a version (no refit)."""
        from . import fitting

        if (
            self.simulation is None or not self.simulation.get("records")
        ) and self.engine is not None:
            # A simulate-only pass regenerates the dataframe but must NOT drop the optimization
            # artifacts `apply_version` just restored — `fit_and_simulate` clears them on a
            # do_fit=False pass — so save and restore them around it.
            saved_optimization = self.optimization
            fitting.fit_and_simulate(self, self.engine, do_fit=False, n_starts=0)
            self.optimization = saved_optimization
        # A payload restored from a stored version may predate the n_estimated or the
        # hardcoded_constants field; stamp them (from the just-applied version's parameter table
        # and model) so fit_result_block can still derive k and report the goodness-of-fit
        # metrics on a restore.
        sim = self.simulation
        if sim is not None:
            if sim.get("n_estimated") is None:
                sim["n_estimated"] = int(self.problem.n_estimated)
            if sim.get("hardcoded_constants") is None:
                sim["hardcoded_constants"] = self.hardcoded_constants()
        return sim

    def hardcoded_constants(self) -> list[str]:
        """The working model's priced literals (see :mod:`literals`) — the addend that turns
        ``n_estimated`` into the ``k`` every score charges."""
        from . import literals

        return literals.hardcoded_constants(self.model.sbml_model)

    # -- nesting check on a fresh commit -----------------------------------
    def _check_nesting_vs_parent(
        self,
        fitsim: dict[str, Any],
        child_model_ant: str,
        is_new_best: bool = False,
    ) -> str | None:
        """Advisory for a commit that gained parameters yet fits WORSE than its parent.

        See :mod:`nesting`: for a NESTED model the ML nll cannot rise when a parameter is added,
        so "k up AND nll up" on a nested child is an INVALID COMPARISON, and on any other child a
        line naming why it is not nested. Advisory only; the commit is never blocked.

        Also maintains ``self.invalid_comparison`` — the session's outstanding invalid
        does-adding-structure-help experiment, which ``fit_summary`` surfaces so a client's finish
        check cannot rest on one. Only a nested child sets it: a client's finish gate repeats the
        re-fit advice, which a not-nested child must not get. Two things clear it: a later commit
        that adds parameters WITHOUT worsening the nll (the experiment, redone properly), and a
        later NEW BEST — the note exists to stop a finish resting on the botched comparison, and a
        session that has since found a strictly better model is resting on that instead. Without
        the second path the flag would stick for the rest of any session that pivots to
        SIMPLIFYING after the botched experiment, since no later commit raises k.

        Called from ``fit`` while holding ``self.lock``, BEFORE the snapshot, so
        ``provenance.current_version`` is still the parent. Best-effort: ANY failure in here —
        the parent read, the child's parameter table, the comparison itself — leaves the payload
        advisory-free rather than failing a commit whose fit has already run.
        """
        from petab.v2.C import LOWER_BOUND, NOMINAL_VALUE, UPPER_BOUND

        from . import nesting
        from .parameters import estimated_ids

        # The model this child DESCENDS FROM, which is not HEAD when the call forked
        # (`make_changes(model_edits=..., base_version=N)`). Comparing k and nll against HEAD
        # there advises on a comparison that never happened — the very defect `derived_from`
        # was added to fix, which this consumer would otherwise still exhibit.
        parent_version = (
            self.derived_from if self.derived_from is not None else self.provenance.current_version
        )
        if parent_version < 0:
            return None
        try:
            parent_meta = self.provenance.get_meta(parent_version)
            parent_df = self.provenance.get_parameter_df(parent_version)
            parent_ids = estimated_ids(parent_df)
            parent = {
                "version": parent_version,
                "n_estimated": len(parent_ids),
                "neg_log_likelihood": parent_meta.get("neg_log_likelihood"),
                "estimated_ids": parent_ids,
                # A version's estimated rows' nominal values are its fitted values.
                "values": parent_df.loc[parent_ids, NOMINAL_VALUE].to_dict(),
                # LAZY: the Antimony is read only once k rose and the nll got worse. The
                # overwhelming majority of commits improve the nll and never resolve it, so the
                # common path stays off model.ant.
                "model": lambda: self.provenance.get_model_antimony(parent_version),
            }
            child_df = self.problem.parameter_df
            child_ids = estimated_ids(child_df)
            child = {
                # Count the estimate=1 ROWS on both sides so the two k's are commensurable;
                # fitsim's n_estimated can also carry analytically-profiled inner parameters the
                # parent's stored table never had, which would fake a rise.
                "n_estimated": len(child_ids),
                "neg_log_likelihood": fitsim.get("neg_log_likelihood"),
                "estimated_ids": child_ids,
                "bounds": {
                    pid: (child_df.at[pid, LOWER_BOUND], child_df.at[pid, UPPER_BOUND])
                    for pid in child_ids
                },
                "model": child_model_ant,
            }
            found = nesting.nesting_advisory(parent, child)
            valid = found is None and nesting.valid_added_parameter_experiment(parent, child)
        except Exception:  # noqa: BLE001 - advisory only; never fail a commit over it
            self.logger.warning("nesting check failed; committing without it", exc_info=True)
            return None
        advisory, invalid = found or (None, False)
        if invalid:
            self.invalid_comparison = advisory
        elif valid or is_new_best:
            self.invalid_comparison = None
        return advisory

    # -- early-stop chain --------------------------------------------------
    def _adopt_checkpoint(self, candidate: Checkpoint, is_new_best: bool) -> bool:
        """Make ``candidate`` the best-score checkpoint if the adoption rule allows; return
        whether it did. The one rule, shared by :meth:`record_fit_progress` and the durable
        resume's replay (:meth:`_seed_progress_from_provenance`), so a restart rebuilds the
        checkpoint the live run kept.

        Only a new best is adopted, and a sigma-GAMED one never OVERWRITES a currently-plausible
        best: the sigma gate would fail the gamed model, so a plausible (would-pass) checkpoint
        must survive a later, better-scoring gamed fit — else the extra search runway could demote
        a passing best to a gamed one and flip a would-pass to a fail. A gamed fit IS still adopted
        when there is no plausible best to protect (best is None, or is itself already gamed), so
        an all-gamed session still records its best-scoring gamed model (the adequacy and sigma
        gates then handle it), keeping the sigma-adequacy gate live rather than dead.
        """
        if not is_new_best or self._declines(candidate.sigma_gamed):
            return False
        self.best = candidate
        return True

    def _declines(self, sigma_gamed: bool) -> bool:
        """Whether the checkpoint keeps its best over a new best whose estimated noise sigma is
        implausible: it does while that best's own noise model is plausible. :meth:`fit` asks it
        before the snapshot, so the ``is_new_best`` a version is stamped with is this decision."""
        return sigma_gamed and self.best is not None and self.best.sigma_plausible

    def record_fit_progress(
        self,
        version: int,
        fitsim: dict[str, Any],
        improvement: dict[str, Any],
    ) -> None:
        """Record one genuine fitted commit as an "op" for the early-stop chain.

        Called after a fit ran. Restores and simulate-only passes are NOT ops and never perturb
        the streak or the checkpoint.
        A new best the checkpoint adopts (see :meth:`_adopt_checkpoint`) resets the non-improving
        streak; any other fit increments it. Since ``is_new_best`` tracks the session's criterion
        (see :func:`fitting.compute_improvement`), the flag, the streak, and the checkpoint stay in
        lock step on one score.

        Two exceptions, each an adopted new best that increments the streak, its op recording why
        (``no_reset``), which the cap's ``stop_reason`` counts. One that gains less than
        ``STREAK_RESET_MIN_SCORE_GAIN`` on ``streak_mark`` is the best but not progress: noise-floor
        wander would otherwise reset the streak forever. One whose ESTIMATED noise sigma collapsed
        is the other: the two implausible tails differ in whether a model can keep improving
        through them. An INFLATED sigma cannot buy a lower score: at the ML optimum sigma_hat is the rms residual, so
        nll = n/2 * log(2*pi*e*sigma^2) rises with sigma, and the inflation is a symptom of the
        misfit, not the source of the gain. A COLLAPSED one can: nll is unbounded below as
        sigma_hat -> 0, checked only by k*ln(n), so a model fitting a small observable group nearly
        exactly gains without limit. The plateau stop never fires on an estimated sigma (see
        :meth:`progress_report`), so the cap is the only bound on that.

        Takes :attr:`counters_lock` itself, under which :meth:`progress_report` reads.
        """
        from . import fitting

        nll = fitsim.get("neg_log_likelihood")
        chi2 = fitsim.get("chi2")
        n = fitsim.get("n_data_points")
        # k = complexity_k (the estimate=1 count plus the hardcoded constants) so this score
        # matches the version meta's — it counts k the same way (fitted_parameters can include
        # inner/fixed params).
        k = fitting.complexity_k(fitsim)
        gof = fitting.goodness_of_fit(
            nll, chi2, k, n, criterion=self.criterion, k_eff=fitsim.get("k_eff")
        )
        score = gof.get("criterion_value")
        gof_p = gof.get("chi2_gof_pvalue")
        is_new_best = bool(improvement.get("is_new_best"))
        # Noise-model plausibility of THIS fit (single source of truth shared with the sigma
        # success gate). Reuse the flags the commit path stashed on fitsim; fall back to computing
        # them for direct callers (e.g. tests) that pass a raw fitsim without them.
        report = fitsim.get("noise_sigma_report")
        sigma_estimated = fitsim.get("noise_sigma_estimated", bool(report))
        sig_inflated = (
            fitsim["noise_sigma_inflated"]
            if "noise_sigma_inflated" in fitsim
            else noise_sigma_inflated(report)
        )
        sig_collapsed = (
            fitsim["noise_sigma_collapsed"]
            if "noise_sigma_collapsed" in fitsim
            else noise_sigma_collapsed(report)
        )

        with self.counters_lock:
            # `score is not None` is NOT enough: a NaN is a float, so a fit whose likelihood
            # came back non-finite would be adopted as the checkpoint and then never beaten
            # (every later `score <= NaN - delta` is False). goodness_of_fit already maps
            # non-finite scores to None; this is the belt-and-braces guard at the write site,
            # so the checkpoint invariant — "score is a real, beatable number" — holds
            # locally no matter how the score was produced.
            usable = score is not None and math.isfinite(score)
            candidate = (
                Checkpoint(
                    version=version,
                    score=score,
                    nll=nll,
                    gof_pvalue=gof_p,
                    sigma_estimated=sigma_estimated,
                    sigma_inflated=sig_inflated,
                    sigma_collapsed=sig_collapsed,
                )
                if usable
                else None
            )
            adopted = candidate is not None and self._adopt_checkpoint(candidate, is_new_best)
            # The adopted new bests that are not progress (see the docstring).
            no_reset = None
            if adopted and candidate.sigma_estimated and candidate.sigma_collapsed:
                no_reset = "a collapsed estimated noise sigma"
            elif (
                adopted
                and self.streak_mark is not None
                and score > self.streak_mark - STREAK_RESET_MIN_SCORE_GAIN
            ):
                no_reset = (
                    f"a {self.criterion.name} gain under {STREAK_RESET_MIN_SCORE_GAIN:g} on the "
                    "best at the last reset"
                )
            if adopted and no_reset is None:
                self.non_improving_streak = 0
                self.streak_mark = score
            else:
                self.non_improving_streak += 1

            self.op_history.append(
                {
                    "version": version,
                    "score": score,
                    # running best AFTER this op
                    "best_score": self.best.score if self.best else None,
                    "gof_pvalue": gof_p,
                    "is_new_best": adopted,
                    **({"no_reset": no_reset} if no_reset else {}),
                }
            )

    def progress_report(self) -> dict[str, Any]:
        """A snapshot of the early-stop chain state — a client polls this to stop a stalled
        session, and each fit result carries its :data:`_PROGRESS_ON_RESULT` keys.

        ``should_stop`` (with a human-readable ``stop_reason``) is the deterministic early-stop
        signal: the hard non-improving-streak cap, or a plateau (the best model shown adequate, by
        a check the reason names, AND the best score gained less than ``PLATEAU_MIN_SCORE_GAIN``
        over the last ``PLATEAU_WINDOW`` ops). When a
        stop fires and the best model is **shown not adequate** (only the streak cap can, a plateau
        needing an adequate best), ``stop_reason`` also carries the issue-#82 nudge to change the
        model's FORM rather than refit or rewrite the same structure. An adequate stop is a success
        — it gets no such nudge (the model fits; there is nothing structural to fix).

        A model whose noise sigmas are ESTIMATED is never shown adequate, so only the streak cap
        ends its session: a fitted sigma absorbs a misfit inside the data-std band without a
        residual flag (bachmann's round-3 run plateau-stopped "adequate" at ΔBIC +62,
        PEtab-MCP#196). Do not restore it as a ratio against the data-driven noise floor: #140's
        was deleted because no single threshold held across problems
        (:func:`fitting._entry_inflated`), and the replicate lack-of-fit arm, its df-aware form,
        has power against a few large misses, not against a diffuse inflation like that run's.
        """
        with self.counters_lock:
            hist = self.op_history
            n_ops = len(hist)
            streak = self.non_improving_streak
            # best-score improvement over the last PLATEAU_WINDOW ops (positive => still falling),
            # on the running best: a gain too small to reset the streak still counts here.
            score_gain_over_window = None
            if n_ops > PLATEAU_WINDOW:
                prev = hist[-PLATEAU_WINDOW - 1]["best_score"]
                cur = hist[-1]["best_score"]
                if prev is not None and cur is not None:
                    score_gain_over_window = prev - cur

            # The check that showed the best model adequate, which a plateau stop needs and names,
            # else None. Only a FIXED-sigma model can be shown adequate: by its chi2 GoF, two-sided
            # (below ADEQUATE_GOF_PVALUE an underfit, above OVERFIT_GOF_PVALUE a fit to the noise).
            # An ESTIMATED-sigma model never is. A residual flag is not read here (PEtab-MCP#247).
            best = self.best
            adequate_by = None
            if (
                best is not None
                and not best.sigma_estimated
                and best.gof_pvalue is not None
                and ADEQUATE_GOF_PVALUE <= best.gof_pvalue <= OVERFIT_GOF_PVALUE
            ):
                adequate_by = f"chi2 GoF p={best.gof_pvalue:.3g} against fixed noise sigmas"
            # Evidence the best model is NOT adequate, which the structural nudge below needs: a
            # fixed-sigma GoF that fails, or an estimated sigma outside its band. A plausible
            # estimated sigma is evidence of neither, and so is a fixed-sigma GoF that could not
            # be computed (no chi2, or n <= k).
            inadequate = (
                best is not None
                and adequate_by is None
                and ((not best.sigma_estimated and best.gof_pvalue is not None) or best.sigma_gamed)
            )
            stop_reason = None
            if streak >= NON_IMPROVING_STREAK_CAP:
                # The new bests inside the streak, by why they did not reset it.
                kept = Counter(op["no_reset"] for op in hist[n_ops - streak :] if "no_reset" in op)
                stop_reason = (
                    f"non-improving-streak cap reached ({streak} consecutive fitted ops with no "
                    f"new best{' that counts as progress' if kept else ''}, cap "
                    f"{NON_IMPROVING_STREAK_CAP}"
                    + "".join(
                        f"; {n} of them were new bests with {why}, which does not"
                        for why, n in kept.items()
                    )
                    + ")."
                )
            elif (
                adequate_by is not None
                and score_gain_over_window is not None
                and score_gain_over_window < PLATEAU_MIN_SCORE_GAIN
            ):
                stop_reason = (
                    f"plateau: best model v{best.version} is adequate ({adequate_by}) and "
                    f"best {self.criterion.name} improved by only "
                    f"{score_gain_over_window:.3g} over the last "
                    f"{PLATEAU_WINDOW} ops (< {PLATEAU_MIN_SCORE_GAIN})."
                )
            # There is deliberately NO hard cap on total fitted ops. One was added and removed: it
            # is not redundant with the streak cap above (that resets on every new best, so 73 of
            # 333 clean sessions passed 20 fits anyway, consuming 43% of all fits, and the streak
            # cap accounted for only 15 of them) -- but it bought compute at the cost of ~1.8% of
            # passes, and it charged that cost UNEVENLY, against exactly the agents that explore
            # longer. That is the wrong trade: a server-side limit silently deciding which agents
            # succeed, in the same family as the reconcile bugs and the sigma gate. The compute
            # case was also weaker than it looked -- of the 32 passers among those 73 long sessions
            # only ~4 crossed the bar after fit 20, so the reclaimed fits were mostly sessions that
            # had already won and were harmlessly polishing. If cost ever needs bounding, bound it
            # in the CLIENT (wall time), not in the scoring path where it silently changes what is
            # measured.
            #
            # A stall with NO checkpoint at all is a different diagnosis: nothing has ever
            # produced a usable score, so the fits are failing to evaluate rather than the
            # model's form failing to fit. Telling the agent to change the STRUCTURE there
            # sends it re-authoring models to escape what is actually an evaluation problem.
            if stop_reason is not None and best is None:
                stop_reason = f"{stop_reason} {_UNSCORED_STALL_ADVICE}"
            # The structural-inadequacy nudge fires only when a stop coincides with a best model
            # shown NOT adequate — appending "the model's FORM cannot fit" to an adequate stop
            # would contradict it, and to an untested one would claim what nothing showed.
            elif stop_reason is not None and inadequate:
                stop_reason = f"{stop_reason} {_STRUCTURAL_PLATEAU_ADVICE}"
                # If the inadequacy is a masking noise model, add the matching noise-side warning so
                # the agent fixes the mechanism instead of leaning on the sigma.
                if best is not None and best.sigma_inflated:
                    stop_reason = f"{stop_reason}{_SIGMA_INFLATED_ADVICE}"
                elif best is not None and best.sigma_collapsed:
                    stop_reason = f"{stop_reason}{_SIGMA_COLLAPSED_ADVICE}"
            return {
                "n_fitted_ops": n_ops,
                "non_improving_streak": streak,
                "streak_cap": NON_IMPROVING_STREAK_CAP,
                "best_version": best.version if best else None,
                "criterion": self.criterion.name,
                "best_score": best.score if best else None,
                "best_gof_pvalue": best.gof_pvalue if best else None,
                "score_improvement_over_last_n": score_gain_over_window,
                "plateau_window": PLATEAU_WINDOW,
                "should_stop": stop_reason is not None,
                "stop_reason": stop_reason,
            }

    @contextlib.contextmanager
    def change(self) -> Iterator[None]:
        """Hold :attr:`lock` for one change and its fit, with :attr:`change_began` set, which
        ``make_changes`` reads to refuse a second change at once
        (:data:`~petab_mcp.errors.FIT_IN_PROGRESS`) rather than queue it behind this one. Its fit
        reports its starts into :attr:`change_progress` (:data:`petab_mcp.multistart.PROGRESS`)."""
        from .multistart import PROGRESS, Progress

        with self.lock:
            self.change_from = self.provenance.last_version
            self.change_progress = Progress()
            token = PROGRESS.set(self.change_progress)
            self.change_began = time.monotonic()
            try:
                yield
            finally:
                self.change_began = None
                PROGRESS.reset(token)

    # -- fit ---------------------------------------------------------------
    def _fit_dedup(self, starts: int) -> tuple[dict[str, Any], str]:
        """Decide what an explicit `fit` should do with the current working model (see :mod:`dedup`).

        The de-dup gates the FIT — the expensive step. Returns ``(decision, fit_key)`` where
        ``decision`` is a :func:`dedup.decide` result and ``fit_key`` identifies this estimation
        problem (structure + setup) for the identical-fit loop guard. This method is READ-ONLY: it
        does not bump the fit-attempt counter — the caller counts an attempt only once it actually
        proceeds to a real fit (so a no-op skip or a rejected fit never inflates the guard). Callers
        hold ``self.lock``.

        ``starts`` is the multistart count this fit would run with: a version fitted with fewer
        does not stand for it (see :func:`dedup.decide`).
        """
        from . import dedup as D
        from .parameters import structure_fingerprint

        structure_hash = structure_fingerprint(self.model.sbml_model, self.problem.parameter_df)
        fit_key = D.reauthor_key(structure_hash, self.problem.parameter_df, starts)
        decision = D.decide(
            structure_hash=structure_hash,
            setup=D.estimation_setup_signature(self.problem.parameter_df),
            starts=starts,
            max_starts=self.max_fit_starts,
            head_version=self.provenance.current_version,
            registry=D.build_registry(self),
            prior_attempts=self._fit_attempts.get(fit_key, 0),  # READ only; incremented on proceed
        )
        return decision, fit_key

    def fit(
        self,
        n_starts: int | None = None,
        justification: str | None = None,
        staged_edits: "list[dict[str, Any]] | None" = None,
    ) -> dict[str, Any]:
        """Fit the current working model, snapshot a version, and score it.

        The single fitting path. ``make_changes`` applies its change to the working model and then
        calls this, so one call is one coherent change, scored once, snapshotted as ONE version.

        ``fit`` scores the assembled model: it fits the free parameters, simulates, and snapshots
        ONE version carrying the fitted parameters + metrics + simulation dataframe, then records
        the early-stop-chain op. ``n_starts`` overrides the server default (and must be >= 1);
        ``justification`` is optional. ``staged_edits`` is what the caller changed, recorded on the
        fit EVENT so the provenance trace names the edit behind each version — it is passed in
        rather than read off session state, because there is no unscored-edit state to read: the
        change and its score are one call. It is deliberately NOT written to the version meta,
        which ``get_version`` returns verbatim to the agent — a staged edit carries the whole
        Antimony string, and shipping every version's copy of it into the context each turn cost
        far more than the label was worth.

        The result also carries the two verdicts this method computes anyway for the version meta
        and the checkpoint: ``residual_structure`` (the compact residual-structure verdict — the
        sign/runs tests plus the replicate lack-of-fit test) and the noise-model verdict
        (``noise_sigma_inflated``/``noise_sigma_collapsed`` + advice), plus — when there is any —
        the ``merge_hints`` naming the cheapest parameter ties the criterion would pay for. Returning them
        saves the caller a separate ``get_fit_summary``/``get_fit_diagnosis`` round-trip;
        ``get_fit_diagnosis`` remains the way to get the detail behind each verdict — the
        per-observable table and worst points, the per-parameter geometry and every priced tie —
        and ``get_fit_summary`` the way for a caller that did NOT run the fit to read the score.

        Before the (expensive) fit it de-duplicates: if the working model is byte-identical to the
        current fitted version the refit is skipped and the stored metrics are returned
        (``committed: false, no_change: true``); a model identical to an EARLIER fitted version is
        refused, naming it, and HEAD stays put — in both cases unless this fit runs MORE
        multistarts than that version did, which is a more thorough search and does run; and the
        same structure fitted too many times is refused, to break an unproductive loop.
        """
        from . import fitting, identifiability
        from .operations import _compile_fit_failure_message

        with self.lock:
            # Validate n_starts BEFORE the de-dup so a rejected value never counts toward the
            # identical-fit loop guard (which is only bumped once we commit to a real fit below).
            n_starts = self._coerce_n_starts(n_starts)

            # The count this fit would actually run with — the de-dup needs it to tell a genuine
            # "fit harder" from a redundant refit (see `_fit_dedup`). Mirrors the same expression
            # in `_auto_fit_and_improvement`.
            starts = self.auto_fit_starts if n_starts is None else n_starts
            decision, fit_key = self._fit_dedup(starts)
            if decision["action"] in ("terminate", "refuse"):
                raise ModelUpdateError(decision["message"])
            if decision["action"] == "skip":
                return self._fit_noop_result(decision, staged_edits)
            warnings = decision.get("warnings", [])
            # We are committing to a real fit of this structure — count it toward the loop guard now
            # (a no-op skip / terminate / rejected n_starts above never reaches here, so only genuine
            # fit attempts are counted).
            self._fit_attempts[fit_key] = self._fit_attempts.get(fit_key, 0) + 1

            # Wrap any AMICI/pyPESTO failure (a model that lints clean but cannot be compiled — a
            # reserved-name codegen clash, an unsupported construct) as a ModelUpdateError, so the
            # tool returns committed=false with an actionable message rather than crashing. Unless
            # the disk is full: the build directory is written to before the model is blamed,
            # since a build whose install step hit the quota fails only at the import after it.
            try:
                fitsim, improvement = self._auto_fit_and_improvement(n_starts)
            except Exception as exc:  # noqa: BLE001 - surface as a clean, actionable rejection
                if isinstance(exc, PetabMcpError):
                    error = disk_full_error(exc) or exc
                else:
                    build_root = getattr(self.engine, "build_root", None)
                    error = disk_full_error(exc, build_root) or ModelUpdateError(
                        _compile_fit_failure_message(exc)
                    )
                if isinstance(error, (InfrastructureError, FitCancelledError)):
                    self._fit_attempts[fit_key] -= 1  # nothing was scored, so a retry is not a loop
                if error is exc:
                    raise
                raise error from exc
            # Starts stopped past their cap (see multistart) are scored as failed ones.
            if killed := fitsim.get("n_starts_killed"):
                warnings = [
                    *warnings,
                    f"Starts stopped past the {self.fit_per_start_timeout:.0f}s per-start "
                    f"cap: {killed} of {starts}. They count as failed starts, and the fit is "
                    f"scored from the other {starts - killed}.",
                ]
            # Most starts out of time: Fides' MAXTIME, or stopped past the cap.
            timed_out = (fitsim.get("exitflag_counts") or {}).get("ExitFlag.MAXTIME", 0) + (
                killed or 0
            )
            if 2 * timed_out > starts:
                warnings = [
                    *warnings,
                    f"{timed_out} of {starts} starts ran out of the "
                    f"{self.fit_per_start_timeout:.0f}s per-start time limit (MAXTIME). More "
                    "starts will not fix that: the limit applies to each start, so added starts "
                    "hit it too. The usual causes are a stiff or badly scaled model, or bounds "
                    "that allow extreme rates.",
                ]
            # A raise that still leaves 1 or 2 starts at the best, fewer than half of those that
            # ran to the end (2 of 2 completed is no needle): more will not help (#287).
            elif (
                starts > self.auto_fit_starts
                and (at_best := fitsim.get("n_starts_at_best")) in (1, 2)
                and 2 * at_best < starts - (killed or 0)
            ):
                warnings = [
                    *warnings,
                    f"{fitsim['n_starts_at_best']} of {starts} starts reached the best value, "
                    f"though this fit ran more than the default {self.auto_fit_starts}: "
                    f"{_NEEDLE_ADVICE}.",
                ]

            staged = list(staged_edits or [])
            nllh = fitsim.get("neg_log_likelihood")
            # Persist the goodness-of-fit p-value in the fit meta + event so a durable resume can
            # restore the best-score checkpoint's adequacy (see _seed_progress_from_provenance);
            # the events otherwise carry the scores/nll but not the p-value.
            gof_pvalue = fitting.goodness_of_fit(
                nllh,
                fitsim.get("chi2"),
                fitting.complexity_k(fitsim),
                fitsim.get("n_data_points"),
                criterion=self.criterion,
            ).get("chi2_gof_pvalue")
            # Residual-structure verdict of THIS fit (the sign/runs tests and the replicate
            # lack-of-fit test), computed ONCE here and stashed on fitsim like the noise flags
            # below. The compact rollup rides on the fit RESULT so the agent doesn't spend a
            # get_fit_diagnosis call on it (issue #113); the bare bool goes into the version meta.
            # Must run BEFORE _snapshot_version so the version meta can carry the flag; safe here
            # since _auto_fit_and_improvement already set self.simulation, which is what the
            # diagnosis reads.
            fitsim["residual_structure"] = residual_structure_rollup(self)
            fitsim["residual_flagged"] = residual_flag(fitsim["residual_structure"])
            # The PARAMETERISATION counterpart of that verdict, on the same reasoning: the cheapest
            # parameter ties the criterion would pay for, priced from the FIM at the optimum this fit just
            # reached (see identifiability.merge_hints). Computed HERE, after
            # _auto_fit_and_improvement, because that is where the fit record it reads was frozen;
            # None (and then absent from the result) when there is no such tie at all, or no FIM. ONLY when
            # an optimizer actually ran: with `auto_fit` off the status is "skipped", the values are
            # whatever was authored, and no record was frozen — not a point a local expansion may be
            # taken at (see identifiability.fit_record).
            # The verdict itself is stashed on the payload beside it (and so snapshotted with the
            # version), because it must outlive this process: a resumed session restores the
            # payload but not the in-memory fit record, and recomputing there would mean naming an
            # AMICI build target from live state — see identifiability.fit_diagnosis_block.
            if fitsim.get("status") == "fit":
                block = identifiability.fit_diagnosis_block(self)
                if block:
                    fitsim[identifiability.STORED_DIAGNOSIS_KEY] = block
                fitsim["merge_hints"] = identifiability.merge_hints(
                    (block or {}).get("diagnosis"), rename=self.text_route
                )
            # Noise-sigma report + flags computed ONCE here and stashed on fitsim, so this event,
            # the checkpoint update in record_fit_progress (below) and the returned fit block all
            # read the SAME verdict instead of recomputing (or disagreeing).
            #
            # The report must be COMPUTED here: fit_and_simulate's payload never carried a
            # `noise_sigma_report` key, so the `fitsim.get(...)` this used to do read None on every
            # real fit — which silently pinned noise_sigma_estimated/inflated/collapsed to False in
            # the event, in the best-score checkpoint and in the plateau-adequacy gate, disabling the
            # implausible-sigma checks entirely. The unit tests inject the key into a synthetic
            # fitsim and call record_fit_progress directly, so they never caught it.
            #
            # Computed BEFORE the snapshot so it can go INTO the version meta: a sigma-plausibility
            # gate on the final model can fail a session whose model beat the BIC bar, and with no
            # per-version report stored there was no way to ask afterwards which versions would
            # have cleared it. That is not hypothetical -- it silently failed 17 of 20 sessions on
            # one problem (best dBIC -28.8, while the single pass was +0.50), and diagnosing it
            # required re-deriving the report from the reference by hand. Storing it makes the gate
            # auditable per version.
            _nsr = fitting.noise_sigma_report(self.problem, fitsim.get("fitted_parameters"))
            # Onto `fitsim` BEFORE the snapshot, not after. `_snapshot_version` evaluates
            # `fitting.build_sim_payload_for_version(sim)` eagerly in its argument list, and that
            # function copies `noise_sigma_report` only IF PRESENT -- so assigning it afterwards
            # (as this did) left every version's simulations.json without one. That silently
            # disabled a sigma gate reading `get_version(...)["simulation"]["noise_sigma_report"]`,
            # so the "score the best model that could pass" scan (#155) degraded to plain best-BIC
            # for every session since it landed: it could never find a gate-clean earlier version
            # because it could never see a report.
            fitsim["noise_sigma_report"] = _nsr
            # The checkpoint's decision, taken here so the version meta, the event and the result
            # say what `best_version` will: a score below the best's with an
            # implausible estimated sigma does not displace a plausible best (see _declines).
            inflated, collapsed = noise_sigma_inflated(_nsr), noise_sigma_collapsed(_nsr)
            if improvement.get("is_new_best") and self._declines(
                bool(_nsr) and (inflated or collapsed)
            ):
                improvement["is_new_best"] = False
                improvement["new_best_declined"] = (
                    f"estimated noise sigma {'collapsed' if collapsed else 'inflated'}: "
                    f"v{self.best.version}, whose noise model is plausible, stays the best"
                )
            # Is this variant actually NESTED in the version it will be compared against? Must run
            # BEFORE _snapshot_version, while provenance.current_version is still the parent. The
            # Antimony is serialized once and handed to both the check (which diffs it against the
            # parent's model.ant) and the snapshot.
            child_model_ant = self.model.to_antimony()
            nesting_advisory = self._check_nesting_vs_parent(
                fitsim, child_model_ant, is_new_best=bool(improvement.get("is_new_best"))
            )
            version = self._snapshot_version(
                model_ant=child_model_ant,
                meta={  # Recorded so a later fit can tell whether it would search HARDER than this
                    # one did (see `_fit_dedup`) — the structure/setup hashes both ignore it.
                    "n_starts": fitsim.get("n_starts"),
                    "n_starts_killed": fitsim.get("n_starts_killed"),
                    "fit_status": fitsim.get("status"),
                    "neg_log_likelihood": nllh,
                    "chi2": fitsim.get("chi2"),
                    "chi2_gof_pvalue": gof_pvalue,
                    "residual_flagged": fitsim["residual_flagged"],
                    "nesting_advisory": nesting_advisory,
                    "noise_sigma_report": _nsr,
                    "noise_sigma_inflated": noise_sigma_inflated(_nsr),
                    "noise_sigma_collapsed": noise_sigma_collapsed(_nsr),
                    **improvement,
                },
                sim=fitsim,
            )
            fitsim["version"] = version
            fitsim["noise_sigma_estimated"] = bool(_nsr)
            fitsim["noise_sigma_inflated"] = noise_sigma_inflated(_nsr)
            fitsim["noise_sigma_collapsed"] = noise_sigma_collapsed(_nsr)
            event = self.provenance.append_event(
                {
                    "type": "fit",
                    "version": version,
                    "justification": justification,
                    "staged_edits": staged,
                    "n_edits": len(staged),
                    "fit_status": fitsim.get("status"),
                    "neg_log_likelihood": nllh,
                    "chi2": fitsim.get("chi2"),
                    "chi2_gof_pvalue": gof_pvalue,
                    "noise_sigma_estimated": fitsim["noise_sigma_estimated"],
                    "noise_sigma_inflated": fitsim["noise_sigma_inflated"],
                    "noise_sigma_collapsed": fitsim["noise_sigma_collapsed"],
                    "nesting_advisory": nesting_advisory,
                    # The OUTSTANDING flag as it stands after this commit (this fit's advisory, an
                    # earlier one this fit did not clear, or None). `nesting_advisory` alone cannot
                    # reconstruct it on a durable resume: it says what THIS commit found, not
                    # whether a later valid experiment has since superseded it. See
                    # `_seed_progress_from_provenance`.
                    "invalid_comparison": self.invalid_comparison,
                    **improvement,
                }
            )
            # Record this as an early-stop-chain op only when a real fit ran (simulate-only passes
            # don't move the streak/checkpoint).
            if fitsim.get("status") == "fit":
                self.record_fit_progress(version, fitsim, improvement)

        block = fitting.fit_result_block(
            fitsim,
            improvement,
            self.problem.parameter_df,
            problem=self.problem,
            criterion=self.criterion,
        )
        # The fit's note on its own score goes to `messages` (see :mod:`messages`), not a field.
        if note := block.pop("objective_note", None):
            warnings = [*warnings, note]
        return {
            "committed": True,
            "version": version,
            "warnings": warnings,
            "event_timestamp": event["timestamp"],
            "progress": {
                k: v
                for k, v in self.progress_report().items()
                if k in _PROGRESS_ON_RESULT and v is not None
            },
            # ABSENT when the variant is a clean nested extension — like `noise_sigma_advice`,
            # this key rides on every fit and is re-sent in the agent's history each turn
            # (PR #23), so it is one line only when there is something to say.
            **({"nesting_advisory": nesting_advisory} if nesting_advisory else {}),
            **fitting.slim_fit_payload(self, block, version=version),
        }

    def _fit_noop_result(
        self, decision: dict[str, Any], staged: "list[dict[str, Any]] | None" = None
    ) -> dict[str, Any]:
        """Short-circuit result for a ``fit`` skipped as byte-identical to the current version.

        No recompile, no fit, no new version — the stored metrics stand. The no-op is logged so the
        avoided churn stays auditable in the provenance trace.

        The score is assembled by the SAME builder a real fit uses, over the session's current
        simulation. That is exact rather than approximate: "no_change" means the working state IS
        the duplicate version's, so its payload is the one to report. Only ``n_starts`` comes from
        the stored meta — it records the search that produced these metrics, and this call ran
        none.
        """
        from . import fitting

        dup_v = decision["duplicate_of"]
        try:
            meta = self.provenance.get_meta(dup_v)
        except Exception:  # noqa: BLE001 - unreadable version: report the no-op without n_starts
            meta = {}
        self.provenance.append_event(
            {
                "type": "fit_noop",
                "duplicate_of_version": dup_v,
                "n_edits": len(staged or []),
            }
        )
        self.logger.info("fit no-op: identical to v%s — skipped recompile+fit+snapshot", dup_v)

        structure = residual_structure_rollup(self)
        flag = residual_flag(structure)
        sim = dict(self.simulation or {})
        sim["residual_structure"] = structure
        # Dropped because a fit RESULT speaks for the call that produced it and this call priced
        # nothing — it did not fit. Popped rather than left alone because `self.simulation` IS the
        # last fit's payload, hints included. `get_fit_summary` reads the STATE instead, and still
        # reports the ties: the structure is unchanged (which is why the no-op fired), so HEAD's fit
        # is still this structure's fit (see identifiability.stored_merge_hints).
        sim.pop("merge_hints", None)
        sim["n_starts"] = meta.get("n_starts")
        sim["n_starts_killed"] = meta.get("n_starts_killed")
        return {
            "committed": False,
            "no_change": True,
            "version": dup_v,
            "duplicate_of_version": dup_v,
            "message": decision.get("message"),
            "warnings": decision.get("warnings", []),
            # Falls back to the stored flag if the tests can't run now; None (never False) when
            # neither is available: "unknown", not a clean bill of health.
            "residual_flagged": (meta.get("residual_flagged") if flag is None else flag),
            **fitting.slim_fit_payload(
                self,
                fitting.fit_result_block(
                    sim,
                    parameter_df=self.problem.parameter_df,
                    problem=self.problem,
                    criterion=self.criterion,
                ),
                version=dup_v,
            ),
        }

    def rollback_to(self, version: int, justification: str | None = None) -> dict[str, Any]:
        """Make a stored version the working state and point HEAD at it.

        No refit and no new version: the target's stored model, parameter table and simulation
        dataframe come back as they are, and HEAD moves to the version that already holds them.
        The move is recorded as an event, so the trace still shows it.

        Every version is a fit. HEAD moving BACKWARDS is what keeps that true — appending a
        rollback version would mint an unfit duplicate of a model that is already stored, and it
        is why ``snapshot_version`` counts from :attr:`ProvenanceStore.last_version` rather than
        from HEAD.
        """
        from . import fitting

        with self.lock:
            self.apply_version(version)
            self.provenance.set_head(version)
            restored = self._restore_or_simulate()
            nllh = restored.get("neg_log_likelihood") if restored else None
            # Verdict of the RESTORED state, so the fit block reads the same everywhere and the
            # agent restoring a prior best sees its structure verdict without a follow-up call.
            # On a COPY: `_restore_or_simulate` hands back `self.simulation` itself, and the
            # verdict is a result-only field that the persisted payload does not carry.
            sim = None
            if restored is not None:
                sim = dict(restored)
                sim["residual_structure"] = residual_structure_rollup(self)
            self.provenance.append_event(
                {
                    "type": "rollback",
                    "rolled_back_to": version,
                    "version": version,
                    "justification": justification,
                    "neg_log_likelihood": nllh,
                }
            )
        return {
            "rolled_back_to": version,
            "version": version,
            **fitting.slim_fit_payload(
                self,
                fitting.fit_result_block(
                    sim,
                    parameter_df=self.problem.parameter_df,
                    problem=self.problem,
                    criterion=self.criterion,
                ),
                version=version,
            ),
        }

    def restore_version(self, version: int, justification: str | None = None) -> dict[str, Any]:
        """Bring a prior version's model + parameters back as the working state (no refit).

        The cheap way to go back to an earlier structure: it restores that version's stored fit
        instead of recompiling and refitting. A version the session's constraints break (one
        committed before a later `set_constraints`) is refused: the working model is always one
        that makes the session's claims.
        """
        if self.constraints is not None:
            from .operations import OperationError

            verdicts = self.constraint_verdicts(int(version), self.constraints)
            broken = [c for c in self.constraints.items if not verdicts[c.id].ok]
            if broken:
                raise OperationError(
                    f"restore_version: version {version} breaks the session's constraints "
                    + ", ".join(f"`{c.id}` ({c.text})" for c in broken)
                    + ", given after it was committed; restore a version that makes them, or "
                    "build one.",
                    cause="constraint_violated",
                )
        return self.rollback_to(version, justification)
