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
    res = _Result([
        _OptimizerResult(history=_Hist([9.0, 3.0, 1.0], [0.0, 0.2, 0.5]),
                         id="0", fval=1.0, time=0.5, n_fval=20, n_grad=20, n_hess=0,
                         exitflag=0, message="converged"),
        _OptimizerResult(history=None, id="1", fval=5.0, time=0.4, n_fval=15),
    ])
    payload = fitting._optimization_payload(res)

    assert payload["n_starts"] == 2
    s0 = payload["starts"][0]
    assert s0["start"] == 0 and s0["fval"] == 1.0 and s0["n_fval"] == 20
    assert s0["message"] == "converged"
    # trace only for the start that had a recorded history, indexed by iteration
    assert [t["fval"] for t in payload["trace"] if t["start"] == 0] == [9.0, 3.0, 1.0]
    assert [t["time"] for t in payload["trace"] if t["start"] == 0] == [0.0, 0.2, 0.5]
    assert all(t["start"] != 1 for t in payload["trace"])  # start 1 had no history


def test_non_finite_fval_becomes_none():
    res = _Result([
        _OptimizerResult(history=_Hist([float("inf"), 2.0]), id="0", fval=float("nan")),
    ])
    payload = fitting._optimization_payload(res)
    assert payload["starts"][0]["fval"] is None  # NaN -> None
    trace_fvals = [t["fval"] for t in payload["trace"]]
    assert trace_fvals == [None, 2.0]  # inf -> None


def test_robust_to_garbage_result():
    # A result whose shape we don't recognise yields empty tables, never an exception.
    assert fitting._optimization_payload(object()) == {"n_starts": 0, "starts": [], "trace": []}


# --- fitted-value extraction: only ESTIMATED params, unscaled to linear -------------------

def test_fitted_values_selects_only_free_params():
    """pyPESTO's x/x_names carry every parameter; `_fitted_values` must keep only the free
    (estimated) ones, so k = len(fitted_parameters) equals n_estimated and the complexity
    scores stay correct. Regression guard for the fixed-params-inflate-k bug."""
    import pytest

    from petab.v1.C import ESTIMATE, NOMINAL_VALUE, PARAMETER_SCALE

    df = _pdf([
        {"parameterId": "k1", "estimate": 1, "nominalValue": 1.0, PARAMETER_SCALE: "log10"},
        {"parameterId": "k2", "estimate": 1, "nominalValue": 1.0, PARAMETER_SCALE: "log10"},
        {"parameterId": "sigma", "estimate": 0, "nominalValue": 0.1, PARAMETER_SCALE: "lin"},
        {"parameterId": "A_init", "estimate": 0, "nominalValue": 1.0, PARAMETER_SCALE: "lin"},
    ])
    # A pyPESTO-style FULL vector (all four params) in scaled space; free indices are k1, k2.
    x_names = ["k1", "k2", "sigma", "A_init"]
    x_values = [0.0, 1.0, 7.0, 9.0]  # log10 -> 10**x; the fixed 7.0/9.0 must be ignored
    fitted = fitting._fitted_values(x_names, x_values, [0, 1], df)

    assert set(fitted) == {"k1", "k2"}          # fixed sigma/A_init excluded
    assert fitted["k1"] == pytest.approx(1.0)   # 10**0
    assert fitted["k2"] == pytest.approx(10.0)  # 10**1
    _ = (ESTIMATE, NOMINAL_VALUE)  # imported for parity with the parameter table shape

    # free_indices=None -> every parameter is treated as free (an all-free problem)
    all_free = fitting._fitted_values(["k1", "k2"], [0.0, 2.0], None,
                                      _pdf([
                                          {"parameterId": "k1", "nominalValue": 1.0,
                                           PARAMETER_SCALE: "log10"},
                                          {"parameterId": "k2", "nominalValue": 1.0,
                                           PARAMETER_SCALE: "log10"}]))
    assert set(all_free) == {"k1", "k2"}


# --- fit cache / fit-skip identity --------------------------------------------------------

def _pdf(rows):
    import pandas as pd

    return pd.DataFrame(rows).set_index("parameterId")


def test_fit_fingerprint_ignores_estimated_start_but_not_fixed():
    base = _pdf([
        {"parameterId": "k", "estimate": 1, "nominalValue": 1.0,
         "lowerBound": 1e-3, "upperBound": 1e3},
        {"parameterId": "s", "estimate": 0, "nominalValue": 0.1,
         "lowerBound": 1e-3, "upperBound": 1e3},
    ])
    key = fitting.fit_fingerprint("model", base, 5)
    assert len(key) == 16

    # the ESTIMATED parameter's starting value is not part of the fit identity
    est_moved = base.copy()
    est_moved.loc["k", "nominalValue"] = 42.0
    assert fitting.fit_fingerprint("model", est_moved, 5) == key

    # but a FIXED parameter's value is (it enters the objective)
    fixed_moved = base.copy()
    fixed_moved.loc["s", "nominalValue"] = 0.2
    assert fitting.fit_fingerprint("model", fixed_moved, 5) != key

    # bounds, n_starts, and the model each change the identity
    bounds = base.copy()
    bounds.loc["k", "upperBound"] = 1e6
    assert fitting.fit_fingerprint("model", bounds, 5) != key
    assert fitting.fit_fingerprint("model", base, 20) != key
    assert fitting.fit_fingerprint("other model", base, 5) != key


def test_fit_cache_roundtrip_and_apply(session):
    key = "deadbeefdeadbeef"
    assert fitting._load_fit_cache(session, key) is None  # empty cache

    fitting._save_fit_cache(session, key, {
        "n_starts": 3,
        "fitted": {"k1": 0.25},
        "neg_log_likelihood": -8.0,
        "optimization": {"n_starts": 3, "starts": [{"start": 0, "fval": 8.0}], "trace": []},
        "result_file": None,
    })
    loaded = fitting._load_fit_cache(session, key)
    assert loaded["fitkey"] == key and loaded["fitted"] == {"k1": 0.25}

    # _apply_fitted writes the cached values into nominalValue (the fit's effect)
    fitting._apply_fitted(session, {"k1": 0.25})
    assert float(session.problem.parameter_df.loc["k1", "nominalValue"]) == 0.25


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
