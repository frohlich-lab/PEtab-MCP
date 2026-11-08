"""``create_server(fix_pointers=False)`` sends the facts and none of the text telling the agent
what to change; on, the server sends what it sent before the switch existed (gemot's
``fix_pointers`` arm)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest
from fastmcp.exceptions import ToolError

from petab_mcp import identifiability as I
from petab_mcp.messages import fix, fix_or, render
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server

YAML_PATH = Path(__file__).parent / "data" / "toy_v2" / "problem.yaml"
_BRACKETS = "⟦⟧⟪⟫"


def _server(tmp_path, pointers: bool):
    return create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / f"prov{int(pointers)}"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
        fix_pointers=pointers,
    )


def test_render_keeps_one_side_and_no_bracket():
    text = "fact." + fix(" Send X" + fix(" now") + ".") + " A " + fix_or("is B", "differs") + "."
    assert render(text, True) == "fact. Send X now. A is B."
    assert render(text, False) == "fact. A differs."


@pytest.mark.parametrize("pointers", [True, False])
def test_what_the_agent_reads_first_carries_no_bracket(tmp_path, pointers):
    """Every description, schema and the instructions, as a client lists them: a bracket left in
    would be read as text, on either side."""
    mcp = _server(tmp_path, pointers)
    tools = {t.name: t for t in asyncio.run(mcp.list_tools())}
    sent = [mcp.instructions] + [f"{t.description}{t.parameters}" for t in tools.values()]
    assert not [s for s in sent if any(b in s for b in _BRACKETS)]
    diagnosis = tools["get_fit_diagnosis"].description
    assert "The expansion is local" in diagnosis  # the fact, either way
    assert ("Remedies are MODEL edits" in diagnosis) is pointers
    assert ("a `merge_hints` row's `take`" in tools["make_changes"].description) is pointers


@pytest.mark.parametrize("pointers", [True, False])
def test_a_skill_keeps_its_facts_and_drops_its_fixes(tmp_path, pointers):
    """Through `get_skill` and the skill's resource alike."""
    mcp = _server(tmp_path, pointers)
    text = asyncio.run(mcp.call_tool("get_skill", {"name": "reading-fits"})).structured_content[
        "result"
    ]
    audit = asyncio.run(mcp.read_resource("petab://skills/independence-audit")).contents[0].content
    for read in (text, audit):
        assert not any(b in read for b in _BRACKETS)
    assert "unidentifiable and its optimum" in text
    assert ("remove its term, or leave it" in text) is pointers
    assert "Which independence assumption" in audit
    assert ("## Ways to meet a requirement" in audit) is pointers


@pytest.mark.parametrize("pointers", [True, False])
def test_a_refusal_keeps_why_and_drops_what_to_send(tmp_path, pointers):
    """The unnamed-reaction refusal: why it is refused stays, the statement to write instead goes."""
    mcp = _server(tmp_path, pointers)
    antimony = YAML_PATH.with_name("model.ant").read_text().replace("r1: ", "", 1)
    res = asyncio.run(
        mcp.call_tool("make_changes", {"antimony": antimony, "justification": "test"})
    ).structured_content
    assert res["applied"] is False and "have no id" in res["error"]
    assert ("Write 'name: A => B; k*A;'" in res["error"]) is pointers
    assert not any(b in str(res) for b in _BRACKETS)


@pytest.mark.parametrize("pointers", [True, False])
def test_a_raised_error_is_rendered_too(tmp_path, pointers):
    """A tool error leaves through every middleware: the brackets must not ride on it."""
    mcp = _server(tmp_path, pointers)

    @mcp.tool
    def boom() -> None:
        raise ToolError("refused." + fix(" Send X instead."))

    with pytest.raises(ToolError) as caught:
        asyncio.run(mcp.call_tool("boom", {}))
    assert str(caught.value) == ("refused. Send X instead." if pointers else "refused.")


def test_off_no_stored_tie_is_offered():
    class _Off:
        fix_pointers = False

    assert I.stored_merge_hints(_Off()) is None


def _off(text: str) -> str:
    out = render(text, False)
    assert not any(b in out for b in _BRACKETS)
    return out


def test_off_the_widening_refusal_names_no_limiting_form(session):
    """The refusal keeps why (unidentifiable, re-pinned) and drops its advice by role (#285)."""
    from petab_mcp import operations as O
    from petab_mcp.operations import OperationError
    from test_operations import _pin_k1_at_upper, _two_negligible_widenings

    _two_negligible_widenings(session)
    _pin_k1_at_upper(session, 1e4)
    with pytest.raises(OperationError) as refused:
        O._reject_repeated_widening(session, session.problem, "k1", 1e-4, 1e5)
    off = _off(str(refused.value))
    assert "is UNIDENTIFIABLE" in off and "values come only from the fit." in off
    assert "LEAVE" not in off and "Do not widen again" not in off


def test_off_a_repeated_refit_is_refused_without_a_next_move():
    from petab_mcp import dedup as D

    d = D.decide(
        structure_hash="X",
        setup=(),
        starts=2,
        max_starts=20,
        head_version=1,
        registry=[],
        prior_attempts=D.MAX_IDENTICAL_REAUTHORS,
    )
    assert _off(d["message"]).endswith(
        "cannot improve the fit. Your best model is kept regardless."
    )


@pytest.mark.parametrize(
    "text,fact,pointer",
    [
        ("fitting:NOISE_SIGMA_INFLATED_ADVICE", "not genuine adequacy here.", "Do NOT widen"),
        ("fitting:NOISE_SIGMA_COLLAPSED_ADVICE", "not genuine adequacy.", "Prefer a simpler"),
        ("nesting:NESTING_ADVISORY_TMPL", "fails to help.", "re-fit with more starts"),
        ("nesting:NOT_NESTED_TMPL", "fails to help.", "add only that structure"),
    ],
)
def test_off_a_flag_keeps_its_fact(text, fact, pointer):
    import importlib

    module, name = text.split(":")
    off = _off(getattr(importlib.import_module(f"petab_mcp.{module}"), name))
    assert fact in off and pointer not in off


def test_off_no_steady_state_names_no_rewrite():
    from petab_mcp.fitting import _no_steady_state

    off = _off(_no_steady_state(["`src`"]))
    assert "has no steady state" in off and "Write the species" not in off


def test_off_a_regime_requirement_names_no_structure():
    """`get_data_regimes`: what the data rule out stays, the structures that would meet it go."""
    import numpy as np
    from test_regimes import _P, _S, _meas, _series

    from petab_mcp.regimes import data_regimes

    t = np.linspace(0, 10, 12)
    r = data_regimes(_S(_P(_meas(_series("obs", "e0", t, 5.0 * t * np.exp(-t))))))
    off = _off(r["observables"]["obs"]["shape"]["requirement"])
    assert off.endswith("cannot produce this at any rate constants")
    assert "Build what the mechanism needs" not in _off(r["how_to_read"])
