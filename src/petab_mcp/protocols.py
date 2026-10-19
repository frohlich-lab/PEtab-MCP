"""An experiment's input protocols: the inputs PEtab cannot express, declared once.

An input a period sets is PEtab v2's own: a condition sets a target's value at the start of a
period, and the problem's tables show it to every route. An input that changes WITHIN a period has
no place there, because a ramp is no value. A problem with one ships ``inputs.tsv`` beside
``problem.yaml``, one row per input. Fujita's EGF ramp, set per experiment by three condition
targets::

    inputId  formula                                                conditionInputs
    EGF      piecewise(EGF_rate*time + EGF_0, time <= EGF_end, 0)   EGF_0;EGF_rate;EGF_end

``formula`` is PEtab math over ``time`` and the condition targets ``conditionInputs`` lists. A
formula that does not read ``time`` is refused, since a period expresses it. The file names no
model, so every route reads the same protocol. The statements route holds the input to it
(:func:`hold`), the Antimony route is shown it (``describe_problem``), and the agent harness's
neural route evaluates it (:meth:`Input.evaluate`). Nothing else declares an input, because each
other copy (an entity-map field, a sentence in a task text) is one that can disagree with this one.

An input is a fact about the experiment, so its formula names only what the experiments set
(:func:`check`). A constant the fit estimates would be a modelling claim the agent can neither see
nor change, which hands it part of the reference's model (Boehm's Epo degradation, Fiedler's
stimulus time constants and Oliveira's transmission steps did, in PEtabAgentBench#6).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import libsbml
import numpy as np
import pandas as pd
import sympy as sp
from numpy.typing import ArrayLike
from petab.v1.math import sympify_petab
from petab.v2.C import TARGET_ID

from .errors import ProtocolError

#: The file, beside ``problem.yaml``.
FILENAME = "inputs.tsv"
_COLUMNS = ("inputId", "formula", "conditionInputs")
_ID = re.compile(r"[A-Za-z_]\w*")
#: PEtab's ``log`` is the natural log; libSBML's L3 parser reads it as log10 unless told.
_PETAB_MATH = libsbml.L3ParserSettings()
_PETAB_MATH.setParseLog(libsbml.L3P_PARSE_LOG_AS_LN)


@dataclass(frozen=True)
class Input:
    """One row of ``inputs.tsv``: ``id`` follows ``formula`` over ``time`` and the condition
    targets ``targets``."""

    id: str
    formula: str
    targets: tuple[str, ...]

    @property
    def rule(self) -> str:
        """The protocol as an Antimony assignment rule."""
        return f"{self.id} := {self.formula}"

    def evaluate(self, values: Mapping[str, ArrayLike], time: ArrayLike) -> np.ndarray:
        """The input at ``time``, each target at its value in ``values``, broadcast together."""
        expr = sympify_petab(self.formula)
        # By name: sympify_petab's symbols carry assumptions a bare `sp.Symbol` does not.
        symbols = {str(s): s for s in expr.free_symbols}
        names = ("time", *self.targets)
        fn = sp.lambdify([symbols.get(n, sp.Symbol(n)) for n in names], expr, modules="numpy")
        args = [
            np.asarray(time, dtype=float),
            *(np.asarray(values[t], float) for t in self.targets),
        ]
        return np.broadcast_to(np.asarray(fn(*args), dtype=float), np.broadcast(*args).shape).copy()


def _names(formula: str) -> set[str]:
    return {str(s) for s in sympify_petab(formula).free_symbols}


def read(directory: str | Path) -> tuple[Input, ...]:
    """The inputs the ``inputs.tsv`` in the problem's ``directory`` declares, none without one;
    refused at the first row that would not be read as written."""
    path = Path(directory) / FILENAME
    if not path.is_file():
        return ()
    table = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if missing := [c for c in _COLUMNS if c not in table.columns]:
        raise ProtocolError(f"{path} has no {', '.join(missing)} column; its header is {_COLUMNS}")
    inputs: list[Input] = []
    for row in table.itertuples(index=False):
        where = f"{path}, input {row.inputId!r}"
        if not _ID.fullmatch(row.inputId):
            raise ProtocolError(f"{where}: an inputId is the id of the quantity it holds")
        if row.inputId in {i.id for i in inputs}:
            raise ProtocolError(f"{where} is declared twice")
        try:
            names = _names(row.formula)
        except ValueError:
            raise ProtocolError(
                f"{where}: {row.formula!r} is no PEtab math; write the protocol as "
                "dose*piecewise(0, time < t_on, 1)"
            ) from None
        if "time" not in names:
            raise ProtocolError(
                f"{where}: {row.formula!r} does not change within a period; set it in the "
                "condition table, whose periods express it"
            )
        targets = tuple(t.strip() for t in row.conditionInputs.split(";") if t.strip())
        if set(targets) != names - {"time"}:
            raise ProtocolError(
                f"{where}: conditionInputs lists {';'.join(targets) or 'nothing'}, but the formula "
                f"reads {';'.join(sorted(names - {'time'})) or 'nothing'}; list what it reads"
            )
        inputs.append(Input(row.inputId, row.formula, targets))
    return tuple(inputs)


def check(inputs: Iterable[Input], problem) -> None:
    """Refuse an input whose formula names anything but ``problem``'s condition targets and
    ``time``."""
    cdf = problem.condition_df
    targets = set() if cdf is None else {str(t) for t in cdf[TARGET_ID].dropna()}
    kinds = {p: "a fixed parameter row" for p in problem.parameter_df.index}
    kinds |= {p: "estimated" for p in problem.x_free_ids}
    named = []
    for item in inputs:
        others = [
            f"{n} ({kinds.get(n, 'in no table, so the fit would estimate it')})"
            for n in sorted(set(item.targets) - targets)
        ]
        if others:
            named.append(f"{item.id!r} names {', '.join(others)}")
    if named:
        raise ProtocolError(
            f"{FILENAME}: an input is a fact about the experiment, so it may name only the "
            f"condition table's targets ({', '.join(sorted(targets)) or 'none'}) and time, but "
            f"{'; '.join(named)}. Give a known value its own condition-table target; a value the "
            "fit finds is a modelling claim, for the agent to write in its own rows."
        )


def load(directory: str | Path, problem) -> tuple[Input, ...]:
    """:func:`read` the inputs in ``directory`` and :func:`check` them against ``problem``, which
    may be a view of the problem there (a held-out set's) that sets more targets."""
    inputs = read(directory)
    check(inputs, problem)
    return inputs


def hold(model: libsbml.Model, species_id: str, item: Input) -> None:
    """Hold species ``species_id`` of ``model`` to ``item``: on the boundary, where a species may
    sit in reactions and under a rule, and set by its protocol's assignment rule. Each target the
    model lacks becomes a parameter, for the condition table to set."""
    model.getSpecies(species_id).setBoundaryCondition(True)
    for target in item.targets:
        if model.getElementBySId(target) is None:
            parameter = model.createParameter()
            parameter.setId(target)
            parameter.setConstant(True)
            parameter.setValue(0.0)
    rule = model.createAssignmentRule()
    rule.setVariable(species_id)
    rule.setMath(libsbml.parseL3FormulaWithSettings(item.formula, _PETAB_MATH))
