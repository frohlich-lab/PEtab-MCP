"""Which complexity penalty ranks the models — BIC by default, the others on request.

Every criterion here has the same deviance form::

    score = 2*nll + penalty(k, n, k_eff)

with ``k`` estimated parameters, ``n`` data points and ``k_eff`` the fit's EFFECTIVE parameter
count (below), and LOWER is better. That shared form is what makes the criterion swappable: the
whole model-building loop (``is_new_best``, the best-score checkpoint, the plateau stop, the de-dup
registry, the merge price) reads ONE number and one effect-size threshold, and those thresholds are
in the same 2·Δnll deviance units under every criterion — so switching the penalty changes what the
session prefers, not how the machinery works.

The penalties differ only in what a parameter costs:

* ``bic``  — ``k*ln(n)``: consistent (picks the true model as n grows), the default because
  over-parameterisation is this project's failure mode.
* ``caic`` — ``k*(ln(n)+1)``: Bozdogan's consistent AIC, BIC plus one more unit per parameter —
  the stricter-than-BIC option, for where BIC still buys structure the data cannot support.
* ``aic``  — ``2k``: efficient rather than consistent, and a flat 2 per parameter — much cheaper
  than BIC for n > 7, so it buys structure BIC would refuse.
* ``aicc`` — AIC with the small-sample correction ``2k(k+1)/(n-k-1)``: what AIC should be whenever
  ``n/k`` is small (say < 40). Undefined at ``n <= k+1``, where it is ``None`` — an
  under-determined fit has no score, which is exactly how a missing score is treated everywhere.
* ``hqc``  — ``2k*ln(ln(n))``: Hannan-Quinn, consistent like BIC but with a penalty that grows
  far more slowly. It sits between AIC and BIC only above ``n = e**e`` (≈ 15.2); BELOW that it is
  the LOOSEST of the three — at n = 10 it charges 1.66 per parameter against AIC's 2.0 and BIC's
  2.30 — so on a small design it buys structure even AIC would refuse.
* ``tic``  — ``2*k_eff``: Takeuchi, AIC with the parameter count replaced by the EFFECTIVE one
  the fit reports (:func:`identifiability.effective_parameter_count`). Equal to ``k`` for a
  correctly specified model, so TIC is AIC there; above it where the model is mis-specified, and
  below it where the data do not determine a parameter — TIC does not charge for a redundant
  parameter, it does not see one, which makes it the wrong criterion for the parsimony failure
  mode and the right one for asking whether the residual structure supports the parameter count.
  ``None`` whenever the fit produced no ``k_eff`` (a resumed session, a version stored before it was
  recorded).

These names and nothing between them: there is no scalar weight on the penalty term, and
``docs/RATIONALE.md`` records why, along with the criteria that were considered for this table and
left out of it.

Resolved once per SESSION, by :func:`resolve` in ``PetabSession.__init__``: the explicit argument
(``--criterion`` on the CLI, ``create_server(criterion=…)``), else the environment
(``PETAB_MCP_CRITERION``), else plain BIC. The session holds the result as ``session.criterion``
and every scorer — ``goodness_of_fit``, the checkpoint, the de-dup registry, the merge price — is
handed that object. There is no module state, so two sessions in one process can rank by
different criteria and nothing can change a session's criterion under it. The first version kept
it behind a module-level ``configure()``, and one caller's ``criterion="aicc"`` then re-scored
every server built after it — invisibly, since the default call passes ``None``.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Callable

from .formatting import finite_float

#: The default: over-parameterisation is the failure mode this project measures, and BIC charges
#: more per parameter than every other penalty here but CAIC at any n a PEtab problem realistically
#: has.
DEFAULT_CRITERION = "bic"

#: ``(k, n, k_eff) -> penalty`` in deviance units, ``None`` where the criterion is undefined.
Penalty = Callable[[int, int, "float | None"], "float | None"]


def _bic(k: int, n: int, k_eff: float | None) -> float | None:
    return k * math.log(n)


def _caic(k: int, n: int, k_eff: float | None) -> float | None:
    return k * (math.log(n) + 1.0)


def _aic(k: int, n: int, k_eff: float | None) -> float | None:
    return 2.0 * k


def _aicc(k: int, n: int, k_eff: float | None) -> float | None:
    # n - k - 1 <= 0 is not a large correction, it is no correction at all: the model has as many
    # parameters as the data can support, and AICc is undefined (it diverges, then flips sign).
    dof = n - k - 1
    return None if dof <= 0 else 2.0 * k + 2.0 * k * (k + 1) / dof


def _hqc(k: int, n: int, k_eff: float | None) -> float | None:
    # ln(ln(n)) is negative below e and undefined at n <= 1 — i.e. the penalty would REWARD
    # parameters. n >= 3 is the smallest count where Hannan-Quinn is a penalty at all.
    return None if n < 3 else 2.0 * k * math.log(math.log(n))


def _tic(k: int, n: int, k_eff: float | None) -> float | None:
    # The only penalty that reads the FIT rather than its dimensions; without the fit's effective
    # count there is nothing to charge, and "nothing" is None, never a fallback to k.
    k_eff = finite_float(k_eff)
    return None if k_eff is None else 2.0 * k_eff


#: name -> complexity penalty in deviance units, ``None`` where that criterion is undefined for
#: the given inputs. The likelihood term ``2*nll`` is added by :meth:`Criterion.value`, so a
#: penalty here is the ONLY thing that distinguishes one criterion from another.
PENALTIES: dict[str, Penalty] = {
    "bic": _bic,
    "aic": _aic,
    "aicc": _aicc,
    "caic": _caic,
    "hqc": _hqc,
    "tic": _tic,
}


@dataclass(frozen=True)
class Criterion:
    """A complexity penalty — the session's notion of "better", in one object.

    Frozen and comparable so ``name`` identifies the scoring rule exactly: a score persisted under
    one criterion must never be ranked against a score computed under another, and
    :meth:`score_of` uses the name to enforce that when reading back a stored fit.
    """

    name: str = DEFAULT_CRITERION
    #: Set in ``__post_init__``; the penalty is looked up once so ``value`` is a plain call.
    _penalty: Penalty = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        name = str(self.name).strip().lower()
        if name not in PENALTIES:
            raise ValueError(
                f"unknown complexity criterion {self.name!r}; "
                f"choose one of {', '.join(sorted(PENALTIES))}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "_penalty", PENALTIES[name])

    @classmethod
    def from_name(cls, name: "str | None") -> "Criterion | None":
        """The criterion a REPORTED name refers to, or ``None`` when it is missing/unrecognised.

        A client scoring a fit reconstructs the scoring rule this way, from the ``criterion`` the
        fit result carries. ``None`` rather than a guess for anything unrecognised, so a caller
        falls back to a criterion it names itself instead of scoring under a different one.
        """
        try:
            return cls(name) if isinstance(name, str) and name.strip() else None
        except ValueError:
            return None

    def penalty(self, k: int | None, n: int | None, k_eff: float | None = None) -> float | None:
        """The complexity charge for ``k`` parameters on ``n`` points (``k_eff`` effective ones),
        or ``None`` when the inputs are missing or this criterion is undefined there."""
        if k is None or n is None or k < 0 or n <= 0:
            return None
        return self._penalty(int(k), int(n), k_eff)

    def value(
        self, nll: float | None, k: int | None, n: int | None, k_eff: float | None = None
    ) -> float | None:
        """``2*nll + penalty``, or ``None`` when any input is missing/non-finite.

        ``k_eff`` is the fit's effective parameter count, read by TIC alone (see the module
        docstring); the other penalties ignore it, so a caller without one passes nothing.

        ``None`` rather than NaN is load-bearing: a NaN is a float and passes every
        ``is not None`` guard downstream, and because NaN compares False against everything, a NaN
        adopted as the best-score checkpoint locks out every later fit for the rest of the session.
        """
        nll = finite_float(nll)
        pen = self.penalty(k, n, k_eff)
        return None if (nll is None or pen is None) else 2.0 * nll + pen

    def parameter_price(self, k: int | None, n: int | None) -> float | None:
        """The neg-log-likelihood a ``k``-th parameter has to buy to pay for itself.

        Half the penalty step from ``k-1`` to ``k`` parameters, because the score charges
        ``2*nll``: a parameter is worth keeping exactly when it lowers nll by more than this.
        Constant in ``k`` for BIC (``ln(n)/2``), CAIC, AIC (``1``) and HQC, but not for AICc,
        whose correction steepens with every parameter already spent.

        ``None`` under TIC, deliberately rather than AIC's 1: what deleting a parameter takes
        off ``k_eff`` is knowable only after the refit, and for the pair a merge hint targets it
        is about zero, because TIC never charged for that parameter. Pricing the merge at 1 would
        promise a saving that is not there.
        """
        if k is None or k < 1:
            return None
        here, before = self.penalty(k, n), self.penalty(k - 1, n)
        return None if (here is None or before is None) else (here - before) / 2.0

    def values(
        self, nll: float | None, k: int | None, n: int | None, k_eff: float | None = None
    ) -> dict[str, float | None]:
        """Every criterion by name, PLUS this criterion's ``name`` and the ``criterion_value``
        the session actually ranks on.

        All of them ride on every fit result and every stored VERSION META, so a finished session
        can be re-read under a criterion it was not run with. Only ``criterion_value`` is
        privileged: it is the one the loop compares, and it equals the entry named by ``criterion``.

        NOT the fit EVENT, which carries only ``bic`` plus the ranked pair (see
        :func:`fitting.compute_improvement`, kept small because it is re-sent in the agent's
        history every turn). The durable resume seeds from EVENTS
        (:meth:`session.PetabSession._seed_progress_from_provenance`), so resuming under a
        criterion the earlier process did not use finds no score and starts fresh rather than
        ranking one criterion's scores against another's.
        """
        out: dict[str, float | None] = {
            name: crit.value(nll, k, n, k_eff) for name, crit in CANONICAL.items()
        }
        out["criterion"] = self.name
        out["criterion_value"] = out[self.name]
        return out

    def score_of(self, record: Mapping) -> float | None:
        """This criterion's score for an already-recorded fit (a version's meta, a fit event).

        By name where the record carries every criterion (a version meta), else off the ranked
        pair when the record was scored under this exact criterion (a fit event, which carries
        only ``bic`` beside that pair). A record with neither is unscored, and callers treat that
        as "no score", not as zero.
        """
        by_name = finite_float(record.get(self.name))
        if by_name is not None:
            return by_name
        return (
            finite_float(record.get("criterion_value"))
            if record.get("criterion") == self.name
            else None
        )


#: One instance per criterion — what :meth:`Criterion.values` reports by name, so a fit carries
#: every score however the server was configured.
CANONICAL: dict[str, Criterion] = {name: Criterion(name) for name in PENALTIES}


def from_env() -> Criterion:
    """The criterion named by ``PETAB_MCP_CRITERION``.

    Falls back to plain BIC on anything unusable rather than failing startup: a typo in an env var
    must not take the server down, and the resolved criterion is reported on every fit,
    so a fallback is visible in the very next result.
    """
    return Criterion.from_name(os.environ.get("PETAB_MCP_CRITERION")) or Criterion()


def resolve(criterion: "Criterion | str | None" = None) -> Criterion:
    """The criterion a session scores by: the explicit argument, else the environment, else BIC.

    Pure — resolves and returns, installs nothing. ``PetabSession.__init__`` calls it once and
    keeps the result, so the resolution happens in exactly one place per session and cannot depend
    on which of several reads happened when.

    RAISES on an unusable explicit argument — a mistake worth reporting at startup — while an
    unusable environment degrades per :func:`from_env`.
    """
    if isinstance(criterion, Criterion):
        return criterion
    return Criterion(criterion) if criterion is not None else from_env()
