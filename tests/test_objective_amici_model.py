"""Tests for ``engine._objective_amici_model`` — reaching the AMICI model behind a pyPESTO
objective whether it's a plain ``AmiciObjective`` (``amici_model`` directly) or an
``AggregatedObjective`` (multi-condition problems, which wrap per-condition sub-objectives and
expose no ``amici_model`` of their own).

Gated on ``pypesto`` only, NOT ``amici``: the aggregate test builds a REAL
``pypesto.AggregatedObjective`` (no compiled model needed — a sentinel stands in for the AMICI
model), so a pyPESTO attribute rename fails HERE instead of silently resurfacing the
"'AggregatedObjective' object has no attribute 'amici_model'" crash in a multi-condition run.
"""

from __future__ import annotations

import pytest

from petab_mcp.engine import _objective_amici_model  # noqa: E402


def test_direct_and_aggregated_fakes():
    """Fakes cover the fallthrough logic: amici_model directly; an aggregate whose sub-objectives
    (skipping one that lacks it) carry it; and the alternate public ``.objectives`` attribute."""
    model = object()
    direct = type("AmiciObj", (), {"amici_model": model})()
    assert _objective_amici_model(direct) is model
    sub = type("Sub", (), {"amici_model": model})()
    agg = type("Agg", (), {"_objectives": [type("NoModel", (), {})(), sub]})()
    assert _objective_amici_model(agg) is model
    agg2 = type("Agg2", (), {"objectives": [sub]})()
    assert _objective_amici_model(agg2) is model


def test_raises_when_absent():
    with pytest.raises(AttributeError):
        _objective_amici_model(type("Empty", (), {})())


def test_real_pypesto_aggregated_objective():
    """The load-bearing test: a REAL ``pypesto.AggregatedObjective``. Pins the attribute the helper
    reads (``_objectives``) to what the installed pyPESTO actually uses, so a rename breaks here
    rather than in a multi-condition production run."""
    from pypesto.objective import AggregatedObjective, Objective

    class _AmiciLike(Objective):
        """A real pyPESTO Objective carrying an ``amici_model`` (as an AmiciObjective would)."""

        def __init__(self, model):
            super().__init__(fun=lambda x: 0.0)
            self.amici_model = model

    sentinel = object()
    agg = AggregatedObjective([_AmiciLike(sentinel), _AmiciLike(sentinel)])
    assert type(agg).__name__ == "AggregatedObjective"
    assert _objective_amici_model(agg) is sentinel

    # A real aggregate whose sub-objectives lack amici_model raises a clean error (the crash path).
    with pytest.raises(AttributeError):
        _objective_amici_model(AggregatedObjective([Objective(fun=lambda x: 0.0)]))
