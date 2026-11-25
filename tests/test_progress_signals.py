"""``create_server(progress_signals=False)``: the agent gets each fit's own numbers and none of the
numbers that rank, compare or judge them (:mod:`petab_mcp.progress_signals`)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from petab_mcp import progress_signals as S
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server

YAML_PATH = Path(__file__).parent / "data" / "toy_v2" / "problem.yaml"

_ANTIMONY = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.5;
  k2 = 0.1;
  r1: A => B; k1 * A;
  r2: B => A; k2 * B;
end
"""

#: A fit's own numbers, which the agent keeps either way.
_OWN = ("neg_log_likelihood", "bic", "criterion_value", "k", "chi2")


def _keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


def test_off_cuts_the_standing_numbers_and_keeps_the_fits_own():
    result = {
        "neg_log_likelihood": 3.0,
        "bic": 8.0,
        "chi2": 4.0,
        "chi2_gof_pvalue": 0.4,
        "reduced_chi2": 1.1,
        "improvement": {"is_new_best": True, "improvement_vs_best_prior": 2.0},
        "progress": {"best_version": 1, "should_stop": False},
        "residual_structure": {
            "flagged_observables": ["obs_a"],
            "low_power": False,
            "summary": "obs_a flagged",
            "replicate_lack_of_fit": {"status": "pass"},
        },
        "message": "Change applied to v0 and saved as v1" + S.signal("; best is v1") + ".",
        "messages": [
            {"level": "warning", "source": "make_changes", "message": S.signal("not the best")},
            {"level": "warning", "source": "fit", "message": "a pinned parameter"},
        ],
        "nesting_advisory": S.signal_or("INVALID COMPARISON vs v0", "nested in v0") + ": re-fit.",
    }
    off = S.strip(result, False)
    assert off == {
        "neg_log_likelihood": 3.0,
        "bic": 8.0,
        "chi2": 4.0,
        "residual_structure": result["residual_structure"],
        "message": "Change applied to v0 and saved as v1.",
        "messages": [{"level": "warning", "source": "fit", "message": "a pinned parameter"}],
        "nesting_advisory": "nested in v0: re-fit.",
    }
    on = S.strip(result, True)
    assert on["message"] == "Change applied to v0 and saved as v1; best is v1."
    assert on["nesting_advisory"] == "INVALID COMPARISON vs v0: re-fit."
    assert on["progress"] == result["progress"] and on["chi2_gof_pvalue"] == 0.4


@pytest.mark.parametrize("signals", [True, False])
def test_end_to_end_through_real_fits(tmp_path, signals):
    """Three fits, the third applied to a version that is not the best, so On sends every standing
    signal: Off sends none of them on any tool the agent reads, the stored record keeps them, and
    get_progress, the client's probe, is whole either way."""
    mcp = create_server(
        str(YAML_PATH),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        progress_signals=signals,
    )

    def call(name, args=None):
        out = asyncio.run(mcp.call_tool(name, args or {})).structured_content
        return out["result"] if set(out) == {"result"} else out

    def k1(lb, ub, **more):
        box = [{"parameter_id": "k1", "lower_bound": lb, "upper_bound": ub}]
        return call("make_changes", {"parameters": box, "n_starts": 2, **more})

    results = [
        k1(1e-3, 2e-3, antimony=_ANTIMONY, justification="k1 boxed in"),
        k1(1e-3, 1e3, justification="let k1 go"),
        k1(1e-3, 3e-3, justification="k1 boxed in again", base_version=0),
    ]
    best = "; best is v{}" if signals else ""
    assert [r["message"] for r in results] == [
        f"Change applied to the loaded problem and saved as v0{best.format(0)}.",
        f"Change applied to v0 and saved as v1{best.format(1)}.",
        f"Change applied to v0 and saved as v2{best.format(1)}.",
    ]
    reads = [
        *results,
        call("list_versions"),
        call("get_version", {"version": 1}),
        call("get_provenance"),
        call("get_fit_summary"),
        call("diff_versions", {"from_version": 0, "to_version": 1}),
        call("wait_for_fit", {"timeout_s": 0}),
    ]
    for read in reads:
        assert bool(_keys(read) & S.KEYS) is signals, read
        assert not any(b in json.dumps(read, ensure_ascii=False) for b in S._BRACKETS)
    for r in results:
        assert all(r[k] is not None for k in _OWN), r
    # HEAD v2 is not the best: a call without `base_version` is refused once only where the agent
    # may be told which is (PEtab-MCP#330).
    refit = call("make_changes", {"n_starts": 3, "justification": "refit v2"})
    assert (refit.get("error_cause") == "implicit_base") is signals
    assert {"summary", "low_power"} <= set(results[2]["residual_structure"])
    assert all(row["criterion_value"] is not None for row in call("list_versions"))

    progress = call("get_progress")
    assert progress["best_version"] == 1 and "should_stop" in progress
    stored = [json.loads(p.read_text()) for p in (tmp_path / "prov").rglob("meta.json")]
    assert any(m.get("is_new_best") for m in stored)

    tools = asyncio.run(mcp.list_tools())
    text = json.dumps(
        [(t.name, t.description, t.parameters) for t in tools if t.name != "get_progress"],
        ensure_ascii=False,
    )
    skill = asyncio.run(mcp.call_tool("get_skill", {"name": "reading-fits"})).content[0].text
    for read in (text, skill):
        assert ("is_new_best" in read and "chi2_gof_pvalue" in read) is signals
        assert not any(b in read for b in S._BRACKETS)
