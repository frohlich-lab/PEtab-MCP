"""``verify_reduction`` (:mod:`petab_mcp.reduction_check`): a statements model, checked through the
server against the model its rows build with no reduction, at its fitted values."""

from __future__ import annotations

import asyncio
import json

from conftest import YAML_PATH

from petab_mcp import reduction_check as RC
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server

#: A becomes B by an enzyme E, which the reduction makes a factor of the step's rate.
_CATALYSED = "conv: A -> B by E | mass_action"


def _server(tmp_path):
    entity_map = tmp_path / "entities.json"
    entity_map.write_text('{"A": {"name": "A"}, "B": {"name": "B"}}')
    return create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        build_root=str(tmp_path / "amici"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
        entity_map=str(entity_map),
    )


def _call(mcp, name, args):
    return asyncio.run(mcp.call_tool(name, args)).structured_content


def _built(tmp_path, **change):
    mcp = _server(tmp_path)
    out = _call(mcp, "make_changes", {"justification": "A becomes B", **change})
    assert out["applied"] is True, out
    return mcp


def test_a_reduced_model_agrees_with_its_unreduced_expansion(tmp_path):
    """The delivered model, at its values, against its rows built unreduced: the same nll and the
    same simulation at every point, written beside the version for whoever reads the run."""
    mcp = _built(tmp_path, statements=[_CATALYSED])
    out = _call(mcp, "verify_reduction", {})
    assert out["status"] == RC.STATUS_OK, out
    assert out["nll"]["reduced"] == out["nll"]["unreduced"] or out["nll_difference"] <= RC.RTOL
    assert out["max_relative_difference"] <= RC.RTOL and out["n_points"] > 0
    stored = json.loads(
        (tmp_path / "prov" / "versions" / "v0" / "reduction_check.json").read_text()
    )
    assert stored["status"] == RC.STATUS_OK and out["file"].endswith("reduction_check.json")


def test_a_model_disagreeing_with_its_expansion_is_a_mismatch(tmp_path, monkeypatch):
    """Should the reduced model ever simulate something else than its rows assert, the check says
    so, with both likelihoods, rather than passing it."""
    mcp = _built(tmp_path, statements=[_CATALYSED])
    unreduced = RC._unreduced_antimony
    monkeypatch.setattr(
        RC,
        "_unreduced_antimony",
        lambda session, record: unreduced(session, record).replace("conv_k*", "conv_k*0.5*", 1),
    )
    out = _call(mcp, "verify_reduction", {})
    assert out["status"] == RC.STATUS_MISMATCH, out
    assert out["nll"]["reduced"] != out["nll"]["unreduced"]
    assert out["max_relative_difference"] > RC.RTOL


def test_a_model_nothing_reduced_is_not_compiled_again(tmp_path):
    """A first-order transition has no catalyst: built unreduced it is the same model, so there
    is nothing to compare and nothing is built."""
    mcp = _built(
        tmp_path,
        statements=[
            {
                "statement": {
                    "type": "PopulationTransition",
                    "source": {"name": "A"},
                    "target": {"name": "B"},
                },
                "policy": "mass_action",
                "prefix": "conv",
            }
        ],
    )
    assert _call(mcp, "verify_reduction", {})["status"] == RC.STATUS_NOT_REDUCED
    assert not (tmp_path / "amici").exists() or not any((tmp_path / "amici").rglob("*.so"))


def test_a_model_written_as_text_has_no_rows_to_check(tmp_path):
    mcp = _built(tmp_path, antimony=(YAML_PATH.parent / "model.ant").read_text())
    assert _call(mcp, "verify_reduction", {})["status"] == RC.STATUS_NOT_STATEMENTS


def test_the_comparison_reads_each_point_against_its_observable():
    """Relative to the point, or to a thousandth of the observable's largest value for a point
    near 0, so an observable that starts at 0 is not a mismatch for its first point's rounding."""
    point = {"observable_id": "o", "experiment_id": "e"}
    reduced = [
        {**point, "time": 0.0, "simulation": 1e-12},
        {**point, "time": 1.0, "simulation": 2.0},
    ]
    full = [{**point, "time": 0.0, "simulation": 2e-12}, {**point, "time": 1.0, "simulation": 2.0}]
    assert RC._compare(reduced, full)["max_relative_difference"] <= RC.RTOL
    full[1]["simulation"] = 2.0 * (1 + 1e-4)
    worst = RC._compare(reduced, full)
    assert worst["max_relative_difference"] > RC.RTOL and worst["worst"]["time"] == 1.0


def test_values_neither_integration_resolves_are_no_mismatch():
    """An observable that stays near 1e-15 differs between two integrations by its own size; under
    the integrations' absolute tolerance that is rounding, not a mismatch."""
    point = {"observable_id": "o", "experiment_id": "e", "time": 1.0}
    assert (
        RC._compare([{**point, "simulation": 8e-16}], [{**point, "simulation": 1.3e-15}])[
            "max_relative_difference"
        ]
        <= RC.RTOL
    )
