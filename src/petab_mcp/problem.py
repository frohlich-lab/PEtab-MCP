"""The PEtab v2 problem a session works on: its SBML model and its tables, as DataFrames.

``petab.v2.Problem`` is the EXCHANGE format — what libpetab lints, AMICI imports and pyPESTO
estimates — and :meth:`WorkingProblem.to_petab` builds one on demand. It is not what the server
edits. Its table views are rebuilt from pydantic objects on every access, so
``problem.parameter_df.loc[p, c] = v`` writes into a temporary and is silently lost; they come
back with ``estimate`` as the strings ``"true"``/``"false"`` and override cells re-printed by
sympy (``0.100000000000000``); and every table read from a file gains an ``index`` column. The
server reads and writes its tables thousands of times per run and fingerprints their exact text,
so it holds them as plain DataFrames — the shape petab v1's ``Problem`` had — and crosses into
libpetab's object model only at :meth:`WorkingProblem.to_petab`.

The tables follow the PEtab v2 schema, with three conventions every consumer may rely on:

* ``parameter_df`` is indexed by ``parameterId``, ``estimate`` is a bool, and
  :data:`PARAMETER_SCALE` is always a column;
* ``measurement_df.experimentId`` is always a string, ``""`` for a measurement simulated without
  an experiment. Not NaN, which is how the file spells it: the server groups and SQL-joins on this
  column everywhere, a NaN key silently drops its rows from a ``groupby`` and never equals itself
  in a join, and "no experiment" is the commonest case there is (every problem without
  conditions);
* priors are not carried. The objective is the likelihood the complexity criterion scores, so a
  prior has nowhere to act, and a table that showed one would promise an effect it does not have.

These are enforced on ASSIGNMENT (``parameter_df``/``observable_df``/``measurement_df`` are
properties whose setters normalise), so a view that swaps a table in gets them too; the
observable table additionally always carries ``noiseDistribution`` and its two placeholder
columns, ``""`` where there are none.

A PEtab v1 problem is upgraded on load (:func:`_from_v1`), table by table and WITHOUT linting:
libpetab's ``petab1to2`` lints the v1 problem first and refuses, but a zero-shot problem ships an
empty SBML whose observables reference nothing yet, which is the input this server exists for.
"""

from __future__ import annotations

import copy
import logging
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from petab import v2
from petab.v2.C import (
    CONDITION_ID,
    ESTIMATE,
    EXPERIMENT_ID,
    LOWER_BOUND,
    NOISE_DISTRIBUTION,
    NOISE_FORMULA,
    NOISE_PLACEHOLDERS,
    NOMINAL_VALUE,
    NORMAL,
    OBSERVABLE_FORMULA,
    OBSERVABLE_ID,
    OBSERVABLE_PLACEHOLDERS,
    PARAMETER_ID,
    PARAMETER_SEPARATOR,
    PRIOR_DISTRIBUTION,
    PRIOR_PARAMETERS,
    TARGET_ID,
    TARGET_VALUE,
    TIME,
    TIME_PREEQUILIBRATION,
    UPPER_BOUND,
)
from petab.v2.models.sbml_model import SbmlModel

logger = logging.getLogger("petab_mcp")

#: The parameter-table column holding the scale a parameter is ESTIMATED on (``lin``, ``log`` or
#: ``log10``). PEtab v2 dropped it from the format, and neither AMICI's v2 simulator nor pyPESTO's
#: v2 importer scales, so the server carries it as an extension column and reparameterises the
#: objective itself (see :mod:`petab_mcp.scaling`). Missing or blank reads as :data:`LIN`.
PARAMETER_SCALE = "parameterScale"
LIN = "lin"
LOG = "log"
LOG10 = "log10"
SCALES = (LIN, LOG, LOG10)

#: Noise distributions whose residual lives in log space; every other one is linear.
_LOG_DISTRIBUTIONS = ("log-normal", "log-laplace")

#: A v1 measurement table's simulation keys: what the upgrade turns into experiments, and how a
#: predicate that accepts a table of either version tells a v1 one apart.
V1_SIMULATION_CONDITION_ID = "simulationConditionId"
V1_PREEQUILIBRATION_CONDITION_ID = "preequilibrationConditionId"

#: The columns a v1 parameter table carries that have no place here (see the module docstring).
_V1_PRIOR_COLUMNS = (
    "objectivePriorType",
    "objectivePriorParameters",
    "initializationPriorType",
    "initializationPriorParameters",
)


def scale(value, scale_str: str):
    """``value`` on estimation scale ``scale_str``. NumPy semantics on purpose: a zero on a log
    scale is ``-inf``, which callers test for, not an exception."""
    if scale_str == LIN:
        return value
    with np.errstate(divide="ignore"):
        if scale_str == LOG:
            return np.log(value)
        if scale_str == LOG10:
            return np.log10(value)
    raise ValueError(f"unknown parameter scale {scale_str!r}")


def unscale(value, scale_str: str):
    """The inverse of :func:`scale`."""
    if scale_str == LIN:
        return value
    if scale_str == LOG:
        return np.exp(value)
    if scale_str == LOG10:
        return np.power(10.0, value)
    raise ValueError(f"unknown parameter scale {scale_str!r}")


def observable_transformation(observable_df, obs_id) -> str:
    """``"log"`` for an observable whose noise distribution is a log one, else ``"lin"`` — the
    coordinate its residuals and noise floor are additive in.

    PEtab v2 folded v1's ``observableTransformation`` into ``noiseDistribution`` and has no log10
    distribution (a v1 log10 observable is upgraded to an exactly equivalent ``log`` one, see
    :func:`_v1_observables`). A blank, absent or unrecognised cell, and an unknown observable,
    read as linear rather than raising: every caller is a residual or a noise yardstick, and a
    raise there loses the point instead of the transformation.
    """
    try:
        if observable_df is None or obs_id not in observable_df.index:
            return LIN
        value = observable_df.loc[obs_id].get(NOISE_DISTRIBUTION)
    except Exception:  # noqa: BLE001 - a duplicate/absent index must not cost the residual
        return LIN
    return LOG if isinstance(value, str) and value.strip().lower() in _LOG_DISTRIBUTIONS else LIN


def as_bool(value) -> bool:
    """A parameter table's ``estimate`` cell as a bool: v2's ``true``/``false``, v1's 0/1, or
    already a bool. Anything else raises — guessing an estimated set is never harmless."""
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)) and not pd.isna(value):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("true", "false", "1", "0"):
        return value.strip().lower() in ("true", "1")
    raise ValueError(f"cannot read estimate value {value!r} as true/false")


def normalize_parameter_df(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` in this module's conventions: indexed by ``parameterId``, ``estimate`` bool,
    :data:`PARAMETER_SCALE` present, no prior columns and no stray ``index`` column. Returns a new
    frame; idempotent, so anything that reads a parameter table from disk passes it through."""
    df = df.copy()
    if df.index.name != PARAMETER_ID:
        if PARAMETER_ID in df.columns:
            df = df.set_index(PARAMETER_ID)
        df.index.name = PARAMETER_ID
    dropped = [c for c in (PRIOR_DISTRIBUTION, PRIOR_PARAMETERS, *_V1_PRIOR_COLUMNS) if c in df]
    if any(df[c].notna().any() and (df[c].astype(str).str.strip() != "").any() for c in dropped):
        logger.warning(
            "the parameter table defines priors (%s); they are dropped, because the objective is "
            "the likelihood the complexity criterion scores",
            ", ".join(dropped),
        )
    df = df.drop(columns=[*dropped, "index"], errors="ignore")
    # Explicit dtypes, because pandas 3 refuses to write a value of another dtype into a column:
    # a table whose bounds are all integers reads as int64 and would refuse the first fitted
    # value, and an EMPTY estimate column would come out float64 and refuse every bool.
    for col in (LOWER_BOUND, UPPER_BOUND, NOMINAL_VALUE):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    if ESTIMATE in df.columns:
        df[ESTIMATE] = pd.Series([as_bool(v) for v in df[ESTIMATE]], index=df.index, dtype=bool)
    if PARAMETER_SCALE in df.columns:
        df[PARAMETER_SCALE] = df[PARAMETER_SCALE].fillna(LIN).astype(str)
    else:
        df[PARAMETER_SCALE] = LIN
    return df


def normalize_observable_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """``df`` indexed by ``observableId``, with ``noiseDistribution`` filled (``normal``, v2's
    default) and the two placeholder columns strings, ``""`` for none. Returns a new frame."""
    if df is None:
        return None
    df = df.drop(columns=["index"], errors="ignore")
    if df.index.name != OBSERVABLE_ID and OBSERVABLE_ID in df.columns:
        df = df.set_index(OBSERVABLE_ID)
    dist = df[NOISE_DISTRIBUTION] if NOISE_DISTRIBUTION in df.columns else None
    df[NOISE_DISTRIBUTION] = (
        NORMAL if dist is None else [v if isinstance(v, str) and v else NORMAL for v in dist]
    )
    for col in (OBSERVABLE_PLACEHOLDERS, NOISE_PLACEHOLDERS):
        cells = df[col] if col in df.columns else pd.Series("", index=df.index)
        df[col] = [v if isinstance(v, str) else "" for v in cells]
    return df


def normalize_measurement_df(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """``df`` with ``experimentId`` a string column, ``""`` where a measurement has no experiment
    (see the module docstring). Returns a new frame."""
    if df is None:
        return None
    df = df.drop(columns=["index"], errors="ignore")
    exp = df[EXPERIMENT_ID] if EXPERIMENT_ID in df.columns else pd.Series("", index=df.index)
    df[EXPERIMENT_ID] = [v if isinstance(v, str) else "" for v in exp]
    return df


class WorkingProblem:
    """The model plus the PEtab v2 tables, mutable in place.

    The design tables (observables, measurements, conditions, experiments, mapping) are fixed for
    the life of a session; the model and ``parameter_df`` are what edits change.
    """

    def __init__(
        self,
        *,
        model: SbmlModel | None = None,
        parameter_df: pd.DataFrame | None = None,
        observable_df: pd.DataFrame | None = None,
        measurement_df: pd.DataFrame | None = None,
        condition_df: pd.DataFrame | None = None,
        experiment_df: pd.DataFrame | None = None,
        mapping_df: pd.DataFrame | None = None,
    ) -> None:
        # {table name: (id of the frame it was built from, libpetab table)} -- see to_petab
        self._petab_tables: dict[str, tuple[int, Any]] = {}
        self.model = model
        self.parameter_df = (
            parameter_df if parameter_df is not None else pd.DataFrame(columns=[PARAMETER_ID])
        )
        self.observable_df = observable_df
        self.measurement_df = measurement_df
        self.condition_df = condition_df
        self.experiment_df = experiment_df
        self.mapping_df = mapping_df

    @property
    def parameter_df(self) -> pd.DataFrame:
        """The parameter table — the stored frame itself, so ``.loc`` writes land."""
        return self._parameter_df

    @parameter_df.setter
    def parameter_df(self, df: pd.DataFrame) -> None:
        self._parameter_df = normalize_parameter_df(df)

    @property
    def observable_df(self) -> pd.DataFrame | None:
        return self._observable_df

    @observable_df.setter
    def observable_df(self, df: pd.DataFrame | None) -> None:
        self._observable_df = normalize_observable_df(df)

    @property
    def measurement_df(self) -> pd.DataFrame | None:
        return self._measurement_df

    @measurement_df.setter
    def measurement_df(self, df: pd.DataFrame | None) -> None:
        self._measurement_df = normalize_measurement_df(df)

    def __copy__(self) -> WorkingProblem:
        """A shallow copy that shares the frames but not the libpetab-table cache, so a view that
        swaps a table in rebuilds only its own."""
        new = object.__new__(WorkingProblem)
        new.__dict__.update(self.__dict__)
        new._petab_tables = {}
        return new

    def __deepcopy__(self, memo) -> WorkingProblem:
        """Frames and model copied; the libpetab-table cache is not (it is rebuilt on demand, and
        copying a large measurement table's pydantic objects costs more than rebuilding them)."""
        new = object.__new__(WorkingProblem)
        memo[id(self)] = new
        for key, value in self.__dict__.items():
            setattr(new, key, {} if key == "_petab_tables" else copy.deepcopy(value, memo))
        return new

    # -- parameter vector ------------------------------------------------------
    @property
    def x_ids(self) -> list[str]:
        return [str(p) for p in self.parameter_df.index]

    @property
    def x_free_ids(self) -> list[str]:
        if ESTIMATE not in self.parameter_df.columns:
            return []
        return [str(p) for p, est in self.parameter_df[ESTIMATE].items() if est]

    @property
    def n_estimated(self) -> int:
        return len(self.x_free_ids)

    # -- loading -----------------------------------------------------------------
    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> WorkingProblem:
        """Load a PEtab problem; a v1 problem is upgraded (see the module docstring)."""
        from petab.versions import get_major_version

        yaml_path = Path(yaml_path)
        config = v2.load_yaml(str(yaml_path))
        if get_major_version(config) == 1:
            return _from_v1(yaml_path)
        return _from_v2(yaml_path, config)

    # -- what a compiled model depends on ----------------------------------------------
    def module_fingerprint(self) -> str:
        """A short hash of everything AMICI's v2 import compiles a model FROM, so two problems
        share a compiled module exactly when it describes both.

        Not the SBML alone, as under v1: the import encodes every experiment and condition as
        events behind an indicator parameter, adds each override id of the measurement table as a
        model parameter, and compiles every parameter the table does not estimate as a constant
        (so a newly estimated one would have no sensitivity). Measurement VALUES and times, and
        every parameter value, are set per simulation and stay out of it.
        """
        import hashlib

        def text(df, columns=None):
            if df is None:
                return ""
            if columns is None:
                return df.to_csv(sep="\t")
            # The distinct combinations only, so two views over different rows of one design share
            # a module.
            df = df[[c for c in columns if c in df.columns]].astype(str).drop_duplicates()
            return df.sort_values(list(df.columns)).to_csv(sep="\t", index=False)

        parts = [
            self.model.to_sbml_str() if self.model is not None else "",
            ",".join(sorted(self.x_free_ids)),
            text(self.observable_df),
            text(
                self.measurement_df,
                [OBSERVABLE_ID, EXPERIMENT_ID, "observableParameters", "noiseParameters"],
            ),
            text(self.condition_df),
            text(self.experiment_df),
            text(self.mapping_df),
        ]
        return hashlib.sha256("\n\0".join(parts).encode("utf-8")).hexdigest()[:16]

    # -- the exchange format ---------------------------------------------------------
    def to_petab(self) -> v2.Problem:
        """This problem as a ``petab.v2.Problem`` — for libpetab's lint, AMICI and pyPESTO.

        The fixed tables are converted once per CONTENT and cached: a large measurement table costs
        seconds to validate into pydantic objects, and nothing downstream mutates them (AMICI
        deep-copies the problem it imports). Keyed on what a frame holds, not on the frame object,
        so a write into one (a held-out view's solved scalings) is never served the stale table.
        The model is SHARED, not copied, and the parameter table is rebuilt on every call, since it
        is what edits change.
        """
        return v2.Problem(
            models=[self.model] if self.model is not None else [],
            parameter_tables=[_parameter_table(self.parameter_df)],
            observable_tables=[self._cached("observables", self.observable_df, _observable_table)],
            measurement_tables=[
                self._cached("measurements", self.measurement_df, _measurement_table)
            ],
            condition_tables=[self._cached("conditions", self.condition_df, _condition_table)],
            experiment_tables=[self._cached("experiments", self.experiment_df, _experiment_table)],
            mapping_tables=[self._cached("mapping", self.mapping_df, _mapping_table)],
        )

    def _cached(self, name: str, df, build):
        key = _content_key(df)
        hit = self._petab_tables.get(name)
        if hit is not None and hit[0] == key:
            return hit[1]
        table = build(df)
        self._petab_tables[name] = (key, table)
        return table


def _content_key(df) -> str | None:
    """A hash of ``df``'s columns, index and values."""
    if df is None:
        return None
    import hashlib

    h = hashlib.sha256(repr((list(df.columns), df.index.name)).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()


# -- libpetab tables -------------------------------------------------------------------
# Built element by element rather than through `XTable.from_df`, which `reset_index`es its input
# (an `index` extra on every element of a RangeIndex frame) and, for measurements, writes into the
# frame it was given.


def _records(df) -> list[dict[str, Any]]:
    """``df``'s rows as dicts, a named index included as a column."""
    if df is None or len(df) == 0:
        return []
    if df.index.name is not None:
        df = df.reset_index()
    return df.to_dict("records")


def _parameter_table(df) -> v2.ParameterTable:
    return v2.ParameterTable([v2.Parameter(**r) for r in _records(df)])


def _observable_table(df) -> v2.ObservableTable:
    return v2.ObservableTable([v2.Observable(**r) for r in _records(df)])


def _measurement_table(df) -> v2.MeasurementTable:
    return v2.MeasurementTable([v2.Measurement(**r) for r in _records(df)])


def _condition_table(df) -> v2.ConditionTable:
    conditions = []
    if df is not None and len(df):
        for cid, sub in df.groupby(CONDITION_ID, sort=False):
            changes = [
                v2.Change(target_id=r[TARGET_ID], target_value=r[TARGET_VALUE])
                for r in sub.to_dict("records")
            ]
            conditions.append(v2.Condition(id=cid, changes=changes))
    return v2.ConditionTable(conditions)


def _experiment_table(df) -> v2.ExperimentTable:
    return v2.ExperimentTable.from_df(df) if df is not None and len(df) else v2.ExperimentTable()


def _mapping_table(df) -> v2.MappingTable:
    return v2.MappingTable([v2.Mapping(**r) for r in _records(df)])


# -- v2 ------------------------------------------------------------------------------------
def _read_tables(paths, base: Path, reader) -> pd.DataFrame | None:
    frames = [reader(str(base / p)) for p in paths or []]
    if not frames:
        return None
    return frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)


def _from_v2(yaml_path: Path, config: dict) -> WorkingProblem:
    """A v2 problem, read as the files have it. The YAML is validated by libpetab's own config
    model, which also resolves the file list."""
    base = yaml_path.parent
    cfg = v2.ProblemConfig(**config, base_path=base, filepath=yaml_path)
    if len(cfg.model_files or {}) > 1:
        raise ValueError("only single-model PEtab problems are supported")
    model = None
    for model_id, info in (cfg.model_files or {}).items():
        if info.language != "sbml":
            raise ValueError(f"only SBML models are supported, not {info.language!r}")
        model = SbmlModel.from_file(base / info.location, model_id=model_id)
        model.rel_path = info.location
    observable_df = _read_tables(cfg.observable_files, base, v2.get_observable_df)
    return WorkingProblem(
        model=model,
        parameter_df=_read_tables(cfg.parameter_files, base, v2.get_parameter_df),
        observable_df=observable_df,
        measurement_df=_read_tables(cfg.measurement_files, base, v2.get_measurement_df),
        condition_df=_read_tables(cfg.condition_files, base, v2.get_condition_df),
        experiment_df=_read_tables(cfg.experiment_files, base, v2.get_experiment_df),
        mapping_df=_read_tables(cfg.mapping_files, base, v2.get_mapping_df),
    )


# -- v1 ------------------------------------------------------------------------------------
def _from_v1(yaml_path: Path) -> WorkingProblem:
    """A v1 problem, upgraded table by table with every likelihood unchanged (see
    :func:`_v1_observables` for the one place that needs care) and its parameter scales kept."""
    from petab import v1

    old = v1.Problem.from_yaml(str(yaml_path))
    condition_df = v1_conditions(old.condition_df)
    experiment_df, measurement_df = v1_measurements(
        old.measurement_df, changing_conditions(condition_df)
    )
    return WorkingProblem(
        model=old.model,
        parameter_df=old.parameter_df,
        observable_df=_v1_observables(old.observable_df),
        measurement_df=measurement_df,
        condition_df=condition_df,
        experiment_df=experiment_df,
        mapping_df=getattr(old, "mapping_df", None),
    )


def v1_conditions(df) -> pd.DataFrame | None:
    """The wide v1 table as v2 changes: one row per non-empty cell, or ``None`` when no cell sets
    anything (a v2 problem without a condition table). ``conditionName`` has no v2 column and is
    dropped; an empty cell set nothing in v1 and sets nothing here."""
    if df is None:
        return None
    wide = df.drop(columns=["conditionName"], errors="ignore")
    long = wide.reset_index().melt(
        id_vars=[CONDITION_ID], var_name=TARGET_ID, value_name=TARGET_VALUE
    )
    long = long.dropna(subset=[TARGET_VALUE]).reset_index(drop=True)
    return long if len(long) else None


def v1_experiment_id(preeq: str, sim: str) -> str:
    """The experiment a v1 (pre-equilibration, simulation) condition pair becomes — libpetab's
    ``petab1to2`` naming, so a problem upgraded there and one upgraded here agree."""
    pre = f"{preeq}_" if preeq else ""
    return f"experiment__{pre}__{sim}"


def changing_conditions(condition_df) -> set[str]:
    """The ids of the conditions in a (long, v2) condition table — every one of which changes
    something, since a v1 condition that set nothing has no row there."""
    return set() if condition_df is None else set(condition_df[CONDITION_ID].astype(str))


def v1_measurements(measurement_df, changing: set[str]) -> tuple[pd.DataFrame | None, pd.DataFrame]:
    """One experiment per measured (pre-equilibration, simulation) pair of a v1 measurement table,
    and the measurement table keyed by it. ``changing`` names the conditions that change anything
    (:func:`changing_conditions`); a v1 condition that set nothing is not one.

    A pair whose conditions change nothing and that does not pre-equilibrate is no experiment at
    all — v2's "simulate the model as is" (``experimentId`` ``""``) — while it is the only such
    pair: ``""`` is one series, and two pairs (two cell lines told apart by their scalings alone)
    are two, so each keeps an experiment of its own. A pre-equilibration with an empty condition is
    still a pre-equilibration, and keeps its period: ``petab1to2`` drops it.
    """
    # v1 had no `experimentId` column, so one there is a free-text annotation (brannmark's names
    # its time-course vs dose-response blocks); it keeps its values under another name.
    mdf = measurement_df.rename(columns={EXPERIMENT_ID: f"v1_{EXPERIMENT_ID}"})
    pre_col, sim_col = V1_PREEQUILIBRATION_CONDITION_ID, V1_SIMULATION_CONDITION_ID
    pre = mdf.pop(pre_col) if pre_col in mdf.columns else pd.Series("", index=mdf.index)
    sim = mdf.pop(sim_col) if sim_col in mdf.columns else pd.Series("", index=mdf.index)
    pre = pre.fillna("").astype(str)
    sim = sim.fillna("").astype(str)

    rows: list[dict[str, Any]] = []
    ids: dict[tuple[str, str], Any] = {}
    pairs = list(dict.fromkeys(zip(pre, sim)))
    as_is = [(p, s) for p, s in pairs if not p and s not in changing]
    for p, s in pairs:
        if as_is == [(p, s)]:
            ids[(p, s)] = ""
            continue
        exp_id = v1_experiment_id(p, s)
        ids[(p, s)] = exp_id
        if p:
            rows.append(
                {
                    EXPERIMENT_ID: exp_id,
                    TIME: TIME_PREEQUILIBRATION,
                    CONDITION_ID: p if p in changing else np.nan,
                }
            )
        rows.append(
            {EXPERIMENT_ID: exp_id, TIME: 0.0, CONDITION_ID: s if s in changing else np.nan}
        )
    # Where v2 files conventionally carry it: right after the observable.
    at = mdf.columns.get_loc(OBSERVABLE_ID) + 1 if OBSERVABLE_ID in mdf.columns else 0
    mdf.insert(at, EXPERIMENT_ID, [ids[k] for k in zip(pre, sim)])
    return (pd.DataFrame(rows) if rows else None), mdf


def _v1_observables(df) -> pd.DataFrame:
    """The v1 observable table in v2 form: ``observableTransformation`` folded into
    ``noiseDistribution``, and the placeholders v1 implied by name declared explicitly.

    v2 has no log10 distribution, and ``petab1to2`` maps v1's ``log10``-``normal`` onto
    ``log-normal`` with the same noise formula — which rescales what sigma means by ln 10 and
    shifts every likelihood by n·ln(ln 10). Here the noise formula is multiplied by ``log(10)``
    instead, which makes the two densities IDENTICAL (residual and Jacobian both carry the factor),
    so an upgraded problem's likelihoods, fitted sigmas and any score recorded against the v1
    problem stay valid.
    """
    out = df.copy()
    trans = out.pop("observableTransformation") if "observableTransformation" in out else None
    dist = (
        out[NOISE_DISTRIBUTION].fillna(NORMAL).astype(str)
        if NOISE_DISTRIBUTION in out
        else pd.Series(NORMAL, index=out.index)
    )
    if trans is not None:
        trans = trans.fillna(LIN).astype(str)
        dist = pd.Series(
            [d if t == LIN else f"log-{d}" for t, d in zip(trans, dist)], index=out.index
        )
        is10 = trans == LOG10
        out.loc[is10, NOISE_FORMULA] = [f"log(10) * ({f})" for f in out.loc[is10, NOISE_FORMULA]]
    out[NOISE_DISTRIBUTION] = dist
    out[OBSERVABLE_PLACEHOLDERS] = [
        _v1_placeholders("observable", oid, f) for oid, f in out[OBSERVABLE_FORMULA].items()
    ]
    out[NOISE_PLACEHOLDERS] = [
        _v1_placeholders("noise", oid, f) for oid, f in out[NOISE_FORMULA].items()
    ]
    return out


def _v1_placeholders(kind: str, obs_id: str, formula) -> str:
    """v1's implicit ``{kind}Parameter{n}_{obs_id}`` placeholders in ``formula``, in ``n`` order —
    numerically, so ten or more line up with their overrides (``petab1to2`` sorts them as text)."""
    if not isinstance(formula, str):
        return ""
    pattern = re.compile(rf"(?<![\w.])({kind}Parameter(\d+)_{re.escape(str(obs_id))})(?!\w)")
    found = {m.group(1): int(m.group(2)) for m in pattern.finditer(formula)}
    return PARAMETER_SEPARATOR.join(sorted(found, key=found.get))


def is_number(token) -> bool:
    """Whether an override cell or one of its ``;``-separated tokens is a number literal."""
    try:
        return math.isfinite(float(token))
    except (TypeError, ValueError):
        return False


def split_overrides(cell) -> list[str | float]:
    """An ``observableParameters``/``noiseParameters`` cell as its overrides, in order: a number
    token as a float, anything else as the (stripped) id it names. An empty cell has none."""
    if cell is None or (isinstance(cell, float) and math.isnan(cell)):
        return []
    if isinstance(cell, (int, float, np.integer, np.floating)):
        return [float(cell)]
    out: list[str | float] = []
    for tok in str(cell).split(PARAMETER_SEPARATOR):
        tok = tok.strip()
        if tok:
            out.append(float(tok) if is_number(tok) else tok)
    return out


def placeholders(observable_df, obs_id, kind: str) -> list[str]:
    """The placeholders observable ``obs_id`` DECLARES for its ``kind`` (``"observable"`` or
    ``"noise"``) formula, in the order the measurement table's overrides fill them."""
    column = OBSERVABLE_PLACEHOLDERS if kind == "observable" else NOISE_PLACEHOLDERS
    try:
        cell = observable_df.loc[obs_id, column]
    except (KeyError, AttributeError):
        return []
    if not isinstance(cell, str):
        return []
    return [p.strip() for p in cell.split(PARAMETER_SEPARATOR) if p.strip()]


def experiment_inputs(problem) -> pd.DataFrame | None:
    """The condition targets in effect while each experiment is SIMULATED, wide: one row per
    experiment id, one column per target, the ``targetValue`` (NaN where the experiment never
    sets that target).

    "In effect while simulated" is the value after the pre-equilibration period and the first
    simulated period have applied their conditions — v2 changes persist, so a target only the
    pre-equilibration sets still holds afterwards, and one the simulated period sets wins. That
    is the v1 simulation condition's row for an upgraded problem, which is what an analysis keyed
    on "the dose of this series" needs. ``None`` when there is no experiment or condition table.
    """
    edf, cdf = problem.experiment_df, problem.condition_df
    if edf is None or cdf is None or not len(edf) or not len(cdf):
        return None
    changes = {
        str(cid): dict(zip(sub[TARGET_ID], sub[TARGET_VALUE]))
        for cid, sub in cdf.groupby(CONDITION_ID, sort=False)
    }
    rows: dict[str, dict[str, Any]] = {}
    for eid, periods in edf.groupby(EXPERIMENT_ID, sort=False):
        times = sorted(float(t) for t in periods[TIME].unique())
        simulated = [t for t in times if t != TIME_PREEQUILIBRATION][:1]
        applied: dict[str, Any] = {}
        for t in [t for t in times if t == TIME_PREEQUILIBRATION] + simulated:
            for cid in periods.loc[periods[TIME] == t, CONDITION_ID].dropna():
                applied.update(changes.get(str(cid), {}))
        rows[str(eid)] = applied
    return pd.DataFrame.from_dict(rows, orient="index")


def placeholder_ids(observable_df) -> set[str]:
    """Every placeholder any observable declares — names the measurement table fills, never
    entities of the model."""
    if observable_df is None:
        return set()
    return {
        p
        for oid in observable_df.index
        for kind in ("observable", "noise")
        for p in placeholders(observable_df, oid, kind)
    }
