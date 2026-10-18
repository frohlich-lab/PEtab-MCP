"""The de-dup on the ``statements`` route, where every rate is an assignment rule.

A model built from statements multiplies each rate law by a rule-defined rate, so changing a rate
leaves every reaction as it was. These drive the in-process MCP server with real fits, as
``test_make_changes.py`` does: the de-dup only skips against a HEAD that was fitted.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server

YAML_PATH = Path(__file__).parent / "data" / "toy_v2" / "problem.yaml"

_ROW = {
    "statement": {"type": "PopulationTransition", "source": {"name": "A"}, "target": {"name": "B"}},
    "prefix": "conv",
}


def _mechanism(policy: str, rate: str) -> dict:
    return {
        "statement_type": "PopulationTransition",
        "policy": policy,
        "based_on": "mass_action",
        "rate": rate,
        "parameters": ["vmax", "K"],
    }


# Both build the reaction `conv_rate * A`; only the rule for `conv_rate` differs.
_SATURATING = _mechanism("saturating", "vmax / (K + source)")
_STEEPER = _mechanism("steeper", "vmax / (K + source * source)")


def _server(tmp_path):
    entity_map = tmp_path / "entities.json"
    entity_map.write_text('{"A": {"name": "A"}, "B": {"name": "B"}}')
    return create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        auto_fit=True,
        entity_map=str(entity_map),
    )


def _call(mcp, name, args):
    return asyncio.run(mcp.call_tool(name, args)).structured_content


def _change(mcp, policy, mechanisms=()):
    args = {"justification": policy, "statements": [{**_ROW, "policy": policy}], "n_starts": 2}
    return _call(mcp, "make_changes", {**args, "mechanisms": list(mechanisms)})


def _rows(mcp):
    return _call(mcp, "get_model", {"format": "statements"})["result"]


def test_a_rate_only_change_is_fitted_not_deduplicated(tmp_path):
    """REGRESSION (rahman r2, schwen r2, oliveira r3, blasi r1): the change came back as a
    duplicate of HEAD and was never fitted."""
    mcp = _server(tmp_path)
    assert _change(mcp, "saturating", [_SATURATING])["committed"] is True
    res = _change(mcp, "steeper", [_STEEPER])
    assert res["committed"] is True and res["version"] == 1, res["message"]


def test_a_duplicate_leaves_no_rows_behind(tmp_path):
    """REGRESSION: a refused duplicate's rows became the model's rows, so the next edit built on
    rows nothing had accepted. The same mechanism under another name builds version 0's model
    exactly: it is refused, and the rows and the run's library stay as they were."""
    mcp = _server(tmp_path)
    _change(mcp, "saturating", [_SATURATING])
    rows = _rows(mcp)

    res = _change(mcp, "renamed", [{**_SATURATING, "policy": "renamed"}])
    assert res["no_change"] is True and res["duplicate_of_version"] == 0

    assert _rows(mcp) == rows
    gap = _change(mcp, "renamed")
    assert gap["applied"] is False and gap["error_cause"] == "library_gap"


def test_a_revert_to_an_earlier_rate_is_not_a_duplicate_of_head(tmp_path):
    """REGRESSION: going back to version 0's rate was refused as a duplicate of HEAD, which holds
    a different one."""
    mcp = _server(tmp_path)
    _change(mcp, "saturating", [_SATURATING])
    _change(mcp, "steeper", [_STEEPER])
    back = _change(mcp, "saturating")
    assert back["committed"] is True and back["version"] == 2, back["message"]
