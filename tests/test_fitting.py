"""Unit tests for pyPESTO result/trace extraction (:func:`petab_mcp.fitting._optimization_payload`).

The extraction is pure — it walks a pyPESTO ``Result``-shaped object and returns plain
records — so it is tested here with a fake result (dict-like ``OptimizerResult`` with a
``history`` attribute, mirroring pyPESTO) without installing pyPESTO/AMICI.
"""

from __future__ import annotations

from petab_mcp import fitting


class _Hist:
    def __init__(self, fvals, times=None):
        self._fvals, self._times = fvals, times or []

    def get_fval_trace(self):
        return self._fvals

    def get_time_trace(self):
        return self._times


class _OptimizerResult(dict):
    """dict-like OptimizerResult carrying a ``.history`` attribute, like pyPESTO's."""

    def __init__(self, history=None, **fields):
        super().__init__(**fields)
        self.history = history


class _Result:
    def __init__(self, starts):
        self.optimize_result = type("OL", (), {"list": starts})()


def test_extracts_starts_and_trace():
    res = _Result(
        [
            _OptimizerResult(
                history=_Hist([9.0, 3.0, 1.0], [0.0, 0.2, 0.5]),
                id="0",
                fval=1.0,
                time=0.5,
                n_fval=20,
                n_grad=20,
                n_hess=0,
                exitflag="ExitFlag.FTOL",
                message="converged",
            ),
            _OptimizerResult(
                history=None,
                id="1",
                fval=5.0,
                time=0.4,
                n_fval=15,
                exitflag="ExitFlag.EXCEEDED_BOUNDARY",
            ),
        ]
    )
    payload = fitting._optimization_payload(res)

    assert payload["n_starts"] == 2
    s0 = payload["starts"][0]
    assert s0["start"] == 0 and s0["fval"] == 1.0 and s0["n_fval"] == 20
    # Per-start exit prose is NOT stored: n_starts copies of `message` answered a histogram
    # question by grepping. The enumerated outcome is aggregated across starts instead.
    assert "message" not in s0 and "exitflag" not in s0
    assert payload["exitflag_counts"] == {"ExitFlag.FTOL": 1, "ExitFlag.EXCEEDED_BOUNDARY": 1}
    # trace only for the start that had a recorded history, indexed by iteration
    assert [t["fval"] for t in payload["trace"] if t["start"] == 0] == [9.0, 3.0, 1.0]
    assert [t["time"] for t in payload["trace"] if t["start"] == 0] == [0.0, 0.2, 0.5]
    assert all(t["start"] != 1 for t in payload["trace"])  # start 1 had no history


def test_non_finite_fval_becomes_none():
    res = _Result(
        [
            _OptimizerResult(history=_Hist([float("inf"), 2.0]), id="0", fval=float("nan")),
        ]
    )
    payload = fitting._optimization_payload(res)
    assert payload["starts"][0]["fval"] is None  # NaN -> None
    trace_fvals = [t["fval"] for t in payload["trace"]]
    assert trace_fvals == [None, 2.0]  # inf -> None


def test_robust_to_garbage_result():
    # A result whose shape we don't recognise yields empty tables, never an exception.
    assert fitting._optimization_payload(object()) == {
        "n_starts": 0,
        "starts": [],
        "trace": [],
        "exitflag_counts": {},
    }


# --- fitted-value extraction: only ESTIMATED params, unscaled to linear -------------------


def test_fitted_values_selects_only_free_params():
    """pyPESTO's x/x_names carry every parameter; `_fitted_values` must keep only the free
    (estimated) ones, so k = len(fitted_parameters) equals n_estimated and the complexity
    scores stay correct. Regression guard for the fixed-params-inflate-k bug."""
    import pytest

    from petab.v2.C import ESTIMATE, NOMINAL_VALUE
    from petab_mcp.problem import PARAMETER_SCALE

    df = _pdf(
        [
            {"parameterId": "k1", "estimate": 1, "nominalValue": 1.0, PARAMETER_SCALE: "log10"},
            {"parameterId": "k2", "estimate": 1, "nominalValue": 1.0, PARAMETER_SCALE: "log10"},
            {"parameterId": "sigma", "estimate": 0, "nominalValue": 0.1, PARAMETER_SCALE: "lin"},
            {"parameterId": "A_init", "estimate": 0, "nominalValue": 1.0, PARAMETER_SCALE: "lin"},
        ]
    )
    # A pyPESTO-style FULL vector (all four params) in scaled space; free indices are k1, k2.
    x_names = ["k1", "k2", "sigma", "A_init"]
    x_values = [0.0, 1.0, 7.0, 9.0]  # log10 -> 10**x; the fixed 7.0/9.0 must be ignored
    fitted = fitting._fitted_values(x_names, x_values, [0, 1], df)

    assert set(fitted) == {"k1", "k2"}  # fixed sigma/A_init excluded
    assert fitted["k1"] == pytest.approx(1.0)  # 10**0
    assert fitted["k2"] == pytest.approx(10.0)  # 10**1
    _ = (ESTIMATE, NOMINAL_VALUE)  # imported for parity with the parameter table shape

    # free_indices=None -> every parameter is treated as free (an all-free problem)
    all_free = fitting._fitted_values(
        ["k1", "k2"],
        [0.0, 2.0],
        None,
        _pdf(
            [
                {"parameterId": "k1", "nominalValue": 1.0, PARAMETER_SCALE: "log10"},
                {"parameterId": "k2", "nominalValue": 1.0, PARAMETER_SCALE: "log10"},
            ]
        ),
    )
    assert set(all_free) == {"k1", "k2"}


# --- fit cache / fit-skip identity --------------------------------------------------------


def _pdf(rows):
    import pandas as pd

    return pd.DataFrame(rows).set_index("parameterId")


def test_default_fit_workers_prefers_slurm(monkeypatch):
    """_default_fit_workers uses SLURM_CPUS_PER_TASK when set (the cores actually granted)."""
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")
    assert fitting._default_fit_workers() == 7


def test_default_fit_workers_falls_back_off_scheduler(monkeypatch):
    """Off a scheduler (or with a junk/<1 SLURM value) it falls back to box-minus-one core."""
    import os

    expected = max(1, (os.cpu_count() or 1) - 1)
    monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
    assert fitting._default_fit_workers() == expected
    for bad in ("0", "-2", "garbage", ""):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", bad)
        assert fitting._default_fit_workers() == expected


def test_fit_n_workers_env_precedence(monkeypatch):
    """FIT_N_WORKERS precedence: PETAB_MCP_FIT_N_WORKERS > SLURM_CPUS_PER_TASK > cpu_count-1."""
    import importlib
    import os

    try:
        monkeypatch.setenv("PETAB_MCP_FIT_N_WORKERS", "3")
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "7")
        importlib.reload(fitting)
        assert fitting.FIT_N_WORKERS == 3  # explicit override wins

        monkeypatch.delenv("PETAB_MCP_FIT_N_WORKERS", raising=False)
        importlib.reload(fitting)
        assert fitting.FIT_N_WORKERS == 7  # then the scheduler grant

        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        importlib.reload(fitting)
        assert fitting.FIT_N_WORKERS == max(1, (os.cpu_count() or 1) - 1)  # then the box default
    finally:
        # Restore the module to the ambient environment so the reloaded module-level
        # constant doesn't leak the patched values into later tests.
        monkeypatch.undo()
        importlib.reload(fitting)


# ------------------------------------------------------- fitted-parameters diff (payload diet)
def test_slim_fitted_parameters_diffs_against_previous_version():
    """The 84-118-entry fitted dict rode on every make_changes/get_fit_summary payload, re-sent
    in history every later turn. It becomes NEW-or-moved->1% vs the previous version's meta;
    full values stay pullable (get_parameters) and archived (meta.json)."""
    from petab_mcp.fitting import _slim_fitted_parameters

    class _Prov:
        """Two reads now: get_meta(version) for the PARENT pointer, then get_meta(parent) for
        its fitted values — the baseline is the recorded parent, not version-1, so a rollback
        does not diff against the branch the session abandoned."""

        def get_meta(self, n):
            if n == 5:
                return {"parent": 4}
            assert n == 4, f"baseline must be the parent (4), got {n}"
            return {"fitted_parameters": {"a": 1.0, "b": 2.0, "c": 3.0}}

    class _S:
        provenance = _Prov()

    out = {
        "version": 5,
        "fitted_parameters": {"a": 1.005, "b": 2.5, "d": 7.0},  # a ~unchanged, b moved, d new
        "parameter_scales": {"a": "log10", "b": "log10", "d": "lin"},
    }
    _slim_fitted_parameters(_S(), out)
    assert out["fitted_parameters"] is None
    assert out["fitted_parameters_changed"] == {"b": 2.5, "d": 7.0}
    assert out["n_parameters_unchanged"] == 1
    assert out["parameter_scales"] == {"b": "log10", "d": "lin"}


def test_slim_fit_payload_slims_a_block_that_carries_no_version_of_its_own():
    """The regression that made the whole payload diet a no-op where it mattered.

    `fit_result_block` emits no `version`, and all three session callers
    (`_fit_result`, the duplicate-commit result, `rollback_to`) put `version` as a SIBLING of the
    block. So `_slim_fitted_parameters` read None, failed its `isinstance(version, int)` guard and
    returned immediately — leaving the 84-118-entry dict on every make_changes result, the
    payload this exists to shrink. It shipped green because the tests above call
    `_slim_fitted_parameters` directly with a hand-built dict that already has `version` in it,
    proving only that the differ works when handed the thing production never hands it.

    The version therefore has to be passed explicitly — and must NOT leak into the returned block,
    or it would collide with the caller's own sibling key."""
    from petab_mcp.fitting import slim_fit_payload

    class _Prov:
        def get_meta(self, n):
            return (
                {"parent": 4} if n == 5 else {"fitted_parameters": {"a": 1.0, "b": 2.0, "c": 3.0}}
            )

    class _S:
        provenance = _Prov()

    block = {
        "fitted_parameters": {"a": 1.005, "b": 2.5, "d": 7.0},
        "parameter_scales": {"a": "log10", "b": "log10", "d": "lin"},
    }
    out = slim_fit_payload(_S(), block, version=5)

    assert out["fitted_parameters"] is None, "the block was not slimmed"
    assert out["fitted_parameters_changed"] == {"b": 2.5, "d": 7.0}
    assert "version" not in out, "version must stay the caller's sibling, not ride in the block"


def test_slim_fitted_parameters_keeps_the_first_fit_whole():
    """version 0 (or no previous fit) has nothing to diff against — the first read is the one
    place the whole vector genuinely informs, and it must stay intact."""
    from petab_mcp.fitting import _slim_fitted_parameters

    out = {"version": 0, "fitted_parameters": {"a": 1.0}, "parameter_scales": {"a": "lin"}}
    _slim_fitted_parameters(object(), out)
    assert out["fitted_parameters"] == {"a": 1.0}
    assert "fitted_parameters_changed" not in out


def test_slim_fitted_parameters_never_degrades_on_failure():
    from petab_mcp.fitting import _slim_fitted_parameters

    class _Boom:
        @property
        def provenance(self):
            raise RuntimeError("store gone")

    out = {"version": 3, "fitted_parameters": {"a": 1.0}, "parameter_scales": {"a": "lin"}}
    _slim_fitted_parameters(_Boom(), out)
    assert out["fitted_parameters"] == {"a": 1.0}  # slimming aid must not break the score payload


def test_exitflag_counts_reaches_the_fit_result_block(monkeypatch):
    """The histogram is PRODUCED in `_optimization_payload` (the `optimization` dict) but READ by
    `fit_result_block` off `fitsim` — two different dicts. Nothing bridged them, so it was None on
    every fit while the per-start `exitflag`/`message` columns it replaced had already been
    removed and the Fides logger dropped to ERROR: converged-vs-truncated was unobtainable from
    the payload, the SQL table and the log simultaneously. Pin every hop."""
    res = _Result(
        [
            _OptimizerResult(history=None, id="0", fval=1.0, exitflag="ExitFlag.FTOL"),
            _OptimizerResult(history=None, id="1", fval=2.0, exitflag="ExitFlag.FTOL"),
            _OptimizerResult(history=None, id="2", fval=9.0, exitflag="ExitFlag.EXCEEDED_BOUNDARY"),
        ]
    )
    payload = fitting._optimization_payload(res)
    expected = {"ExitFlag.FTOL": 2, "ExitFlag.EXCEEDED_BOUNDARY": 1}
    assert payload["exitflag_counts"] == expected
    # survives per version, or a restored version cannot answer "optimiser- or structure-limited?"
    assert fitting.build_optimization_payload_for_version(payload)["exitflag_counts"] == expected
    # and reaches the block the agent and a scorer actually read
    fitsim = {
        "status": "fit",
        "neg_log_likelihood": -5.0,
        "chi2": 4.0,
        "n_data_points": 10,
        "n_estimated": 2,
        "n_starts": 3,
        "exitflag_counts": expected,
    }
    assert fitting.fit_result_block(fitsim, None, None, None)["exitflag_counts"] == expected


def _mdf(rows):
    import pandas as pd

    return pd.DataFrame(rows)


def _problem(mdf):
    class _P:
        measurement_df = mdf

    return _P()


def test_numeric_per_timepoint_sigmas_do_not_force_flattening():
    """AMICI writes a numeric sigma straight into the ExpData, so per-timepoint numbers are
    already exact. MEASURED on a published problem: the nominal llh is bit-identical flattened and
    unflattened (-190.96512533983716). Flattening them would be pure cost -- it multiplies the
    observable count, and the same predicate gates the observable merge."""
    from petab_mcp.fitting import needs_flattening

    mdf = _mdf(
        [
            {
                "observableId": "a",
                "simulationConditionId": "c",
                "time": 0.0,
                "noiseParameters": 0.01,
            },
            {
                "observableId": "a",
                "simulationConditionId": "c",
                "time": 1.0,
                "noiseParameters": 0.02,
            },
        ]
    )
    assert needs_flattening(_problem(mdf)) is False


def test_numeric_per_timepoint_observable_parameters_do_force_flattening():
    """These DO become mapped parameters, one value per condition, so a number that varies
    between timepoints of one condition has nowhere to live."""
    from petab_mcp.fitting import needs_flattening

    mdf = _mdf(
        [
            {
                "observableId": "a",
                "simulationConditionId": "c",
                "time": 0.0,
                "observableParameters": 1.0,
            },
            {
                "observableId": "a",
                "simulationConditionId": "c",
                "time": 1.0,
                "observableParameters": 2.0,
            },
        ]
    )
    assert needs_flattening(_problem(mdf)) is True


def test_overrides_that_vary_only_across_conditions_are_taken_natively():
    """AMICI's parameter mapping is per condition, so this is not a flattening case."""
    from petab_mcp.fitting import needs_flattening

    mdf = _mdf(
        [
            {
                "observableId": "a",
                "simulationConditionId": "c1",
                "time": 0.0,
                "observableParameters": "s1",
            },
            {
                "observableId": "a",
                "simulationConditionId": "c2",
                "time": 0.0,
                "observableParameters": "s2",
            },
        ]
    )
    assert needs_flattening(_problem(mdf)) is False
