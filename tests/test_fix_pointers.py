"""``create_server(fix_pointers=False)`` withholds the numbers that point the agent at what to
change or where the fit is off, and sends the rest as before (gemot's ``fix_pointers`` arm)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from petab_mcp import data_access as D
from petab_mcp import withheld
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server

YAML_PATH = Path(__file__).parent / "data" / "toy_v2" / "problem.yaml"


def _keys(value) -> set[str]:
    if isinstance(value, dict):
        return set(value) | {k for v in value.values() for k in _keys(v)}
    if isinstance(value, list):
        return {k for v in value for k in _keys(v)}
    return set()


def _flagged_rollup(flagged: list[str]) -> dict:
    """A residual verdict as a fit result carries it, flagging ``flagged`` and a replicated point,
    built by the server's own summary builders."""
    verdict = {
        "flagged": ["obs_a[e1, t=2]"],
        "n_points": 4,
        "points": [{"label": "obs_a[e1, t=2]", "z": 3.2, "n": 2, "p_holm": 0.001}],
        "min_p_holm": 0.001,
        "alpha": 0.01,
    }
    lof = {**verdict, "status": "flagged", "available": True, "max_abs_z": 3.2}
    lof["advice"] = D._replicate_lof_advice(verdict)
    rollup = {
        "flagged_observables": flagged,
        "low_power": False,
        "summary": D._structure_summary(flagged, False, lof),
        "replicate_lack_of_fit": lof,
        "sigma_heterogeneity": [{"parameter_id": "sd_a", "p": 0.001, "worst_series": "e1"}],
    }
    return D.residual_structure_on_fit(rollup)


def test_a_verdict_keeps_whether_it_flags_and_drops_where():
    for flagged in (["obs_a", "obs_b"], ["obs_a", "obs_b", "obs_c", "obs_d", "obs_e"]):
        cut = withheld.strip({"bic": 12.5, "residual_structure": _flagged_rollup(flagged)})
        rs = cut["residual_structure"]
        assert cut["bic"] == 12.5 and rs["n_flagged_observables"] == len(flagged)
        assert f"residual structure in {len(flagged)} observables (" in rs["summary"]
        assert "Holm p 0.001" in rs["summary"]  # the test's own p stays
        assert not [o for o in (*flagged, "obs_a[e1") if o in json.dumps(cut)]
        assert not {"flagged_observables", "sigma_heterogeneity", "points", "flagged"} & _keys(cut)


def test_a_ties_payoff_goes_and_its_cost_stays():
    """The identifiability facts stay: what a tie costs in fit and how far it moves a parameter."""
    diagnosis = {
        "candidates": [{"parameter_id": "k2", "at_bound": True}],
        "merge_candidates": {
            "ties": [
                {
                    "parameters": ["k1", "k2"],
                    "dnll": 0.01,
                    "dscore": -4.6,
                    "merge_improves_score": True,
                    "move_scaled": 0.2,
                }
            ],
            "n_merges_improving_score": 1,
        },
    }
    cut = withheld.strip({"merge_hints": [{"take": {"op": "rename"}}], "diagnosis": diagnosis})
    assert cut == {
        "diagnosis": {
            "merge_candidates": {
                "ties": [{"parameters": ["k1", "k2"], "dnll": 0.01, "move_scaled": 0.2}]
            }
        }
    }


def _servers(tmp_path, *, auto_fit=False):
    return {
        pointers: create_server(
            str(YAML_PATH),
            provenance_dir=str(tmp_path / f"prov{int(pointers)}"),
            resolver=StubReferenceResolver(),
            auto_fit=auto_fit,
            fix_pointers=pointers,
        )
        for pointers in (True, False)
    }


def test_off_every_result_leaves_without_them_and_on_untouched(tmp_path):
    """The structured content and its JSON text block alike; with the switch on nothing is cut."""
    payload = {"bic": 1.0, "merge_hints": [{"parameters": ["k1", "k2"]}], "worst_points": [{}]}
    for pointers, mcp in _servers(tmp_path).items():

        @mcp.tool
        def scored() -> dict:
            return payload

        result = asyncio.run(mcp.call_tool("scored", {}))
        sent = (result.structured_content, json.loads(result.content[0].text))
        assert all(s == (payload if pointers else {"bic": 1.0}) for s in sent)


def test_off_a_real_fit_and_its_reads_carry_none_of_them(tmp_path):
    """make_changes' fit, get_fit_summary and get_fit_diagnosis on the toy problem."""
    mcp = _servers(tmp_path, auto_fit=True)[False]
    antimony = YAML_PATH.with_name("model.ant").read_text().replace("k2 = 0.1", "k2 = 0.2")

    async def go():
        fit = await mcp.call_tool("make_changes", {"antimony": antimony, "justification": "t"})
        summary = await mcp.call_tool("get_fit_summary", {})
        diagnosis = await mcp.call_tool("get_fit_diagnosis", {"sections": "all"})
        return [r.structured_content for r in (fit, summary, diagnosis)]

    sent = asyncio.run(go())
    assert sent[0]["committed"] is True and "bic" in sent[1]
    assert not (_keys(sent) & (withheld._CUT | {"flagged_observables"}))
    assert "n_flagged_observables" in sent[1]["residual_structure"]
