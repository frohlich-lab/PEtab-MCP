"""What a version records about WHERE IT CAME FROM: its true parent, and the caller's own text.

Both exist because three separate investigations broke on their absence:

* ``parent`` defaulted to HEAD and the commit path never passed ``base_version`` down, so a fork
  (patch version N's stored text while HEAD is elsewhere) was recorded as a child of HEAD.
  Measured on 30 diff-edit sessions: 207 of 311 ``base_version`` calls pointed at something older
  than HEAD, and 30/30 version trees still looked perfectly linear. Everything that reasons
  parent-to-child then compares against a model the child never came from.
* ``model.ant`` is the BUILD OUTPUT (reconciled, libantimony re-rendered), so lineage cannot be
  recovered from it after the fact — inferring parents by text distance mis-assigned 43% on
  sessions that could not even fork. ``model_source.ant`` is what the caller authored or edited
  — and with structured `model_edits` it IS a source of truth: operations apply to it, so a
  child's source is its parent's source plus that call's operations.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from conftest import DATA_DIR

from petab_mcp.references import StubReferenceResolver


def _server(tmp_path, tag="prov"):
    from petab_mcp.server import create_server

    work = tmp_path / f"toy_{tag}"
    work.mkdir()
    for f in DATA_DIR.iterdir():
        shutil.copy(f, work / f.name)
    return create_server(
        str(work / "problem.yaml"),
        provenance_dir=str(tmp_path / f"prov_{tag}"),
        build_root=str(tmp_path / f"amici_{tag}"),
        resolver=StubReferenceResolver(),
        auto_fit=True,
    ), tmp_path / f"prov_{tag}"


def _call(mcp, name, args):
    return asyncio.run(mcp.call_tool(name, args)).structured_content


@pytest.fixture
def served(tmp_path):
    return _server(tmp_path)


def _start_objective(result: dict) -> float:
    """The objective AT the fit's starting point, before the optimizer moves.

    Lives under `improvement` (`compute_improvement`'s dict, nested by `fitting.py`), not at the
    top level — and that block is `improvement or None`, so a simulate-only commit has no starting
    objective at all. Asserted rather than indexed so a payload change fails with THAT as the
    message instead of a bare KeyError in a test about warm starts."""
    imp = result.get("improvement")
    assert imp, (
        f"the commit response carries no `improvement` block (keys: {sorted(result)}) — "
        "without it there is no observable for which point a fork started from"
    )
    assert "objective_before" in imp, (
        f"`improvement` no longer carries `objective_before` (keys: {sorted(imp)})"
    )
    val = imp["objective_before"]
    assert val is not None, "objective_before is None: the starting point was not evaluated"
    return val


def _meta(prov: Path, v: int) -> dict:
    return json.loads((prov / "versions" / f"v{v}" / "meta.json").read_text())


def test_a_fork_is_recorded_as_a_child_of_the_version_it_patched(served):
    """The bug this closes: v2 patched from v0 while HEAD was v1 was recorded as a child of v1,
    so every parent-to-child comparison used a model v2 never came from."""
    mcp, prov = served
    r0 = _call(
        mcp,
        "make_changes",
        {
            "justification": "v0",
            "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "(k1/2 + k1/2)*A"}],
        },
    )
    r1 = _call(
        mcp,
        "make_changes",
        {
            "justification": "v1",
            "model_edits": [{"op": "set_rate_law", "id": "r2", "antimony": "(k2/2 + k2/2)*B"}],
        },
    )
    assert (r0["version"], r1["version"]) == (0, 1)
    # HEAD is v1; fork from v0.
    r2 = _call(
        mcp,
        "make_changes",
        {
            "justification": "fork from v0",
            "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "(k1/3 + 2*k1/3)*A"}],
            "base_version": 0,
        },
    )
    assert r2["committed"] is True
    assert _meta(prov, r2["version"])["parent"] == 0, "a fork is still recorded as a child of HEAD"
    assert _meta(prov, 1)["parent"] == 0  # the ordinary chain is unaffected


def test_an_ordinary_commit_still_descends_from_head(served):
    """The fork parent must not leak: `derived_from` is per-call state, so the NEXT ordinary
    commit has to go back to HEAD or every later version inherits a stale fork parent. The fork
    writes r1 in a form neither version has: v0's own model again is refused as v0."""
    mcp, prov = served

    def op(justification, rid, law, **kw):
        return _call(
            mcp,
            "make_changes",
            {
                "justification": justification,
                "model_edits": [{"op": "set_rate_law", "id": rid, "antimony": law}],
                **kw,
            },
        )

    op("v0", "r1", "(k1*1.0)*A")
    op("v1", "r2", "(k2*1.0)*B")
    forked = op("fork", "r1", "(k1/2 + k1/2)*A", base_version=0)
    after = op("ordinary commit after a fork", "r2", "(k2*1.00)*B")
    assert forked["committed"] is True and after["committed"] is True, (forked, after)
    assert _meta(prov, forked["version"])["parent"] == 0
    assert _meta(prov, after["version"])["parent"] == forked["version"], (
        "an ordinary commit inherited the fork's base instead of HEAD"
    )


def test_the_callers_own_text_is_stored_beside_the_built_model(served):
    """`model.ant` is the reconciled, re-rendered build output; `model_source.ant` is what the
    caller submitted. Keeping both is what makes lineage checkable after the fact."""
    mcp, prov = served
    res = _call(
        mcp,
        "make_changes",
        {
            "justification": "authored wholesale",
            "antimony": _call(mcp, "get_model", {})["result"].replace("k1*A", "(k1 + 0.0)*A"),
        },
    )
    assert res["committed"] is True
    vdir = prov / "versions" / f"v{res['version']}"
    built = (vdir / "model.ant").read_text()
    source = (vdir / "model_source.ant").read_text()
    assert "(k1 + 0.0)*A" in source  # the caller's text, verbatim
    assert source != built, "source and build output are identical — the render did nothing?"
    # The build output carries what the reconcile added; the source does not have to.
    assert "const" in built


def test_a_parameters_only_commit_INHERITS_the_source_it_did_not_change(served):
    """A bounds-only commit submits no model text, but the version still HAS a model, so it still
    has a source: the one it inherited.

    This used to store nothing, on the reasoning that there was nothing to record. The hole that
    left: a resume or rollback landing on such a version found no `model_source.ant` and fell back
    to the reconciled RENDER as the branch base, so the next batch of operations edited render
    text, `child.source == parent.source + ops` quietly stopped holding, and nothing in the
    payload marked where. The original worry — an EMPTY file reading as "the agent authored an
    empty model" — is still guarded, because an inherited source is never empty.
    """
    mcp, prov = served
    base = _call(
        mcp,
        "make_changes",
        {
            "justification": "seed",
            "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "(k1*1.0)*A"}],
        },
    )
    res = _call(
        mcp,
        "make_changes",
        {
            "justification": "bounds only",
            "parameters": [{"parameter_id": "k1", "lower_bound": 1e-4, "upper_bound": 1e4}],
        },
    )
    assert res["applied"] is True
    base_src = (prov / "versions" / f"v{base['version']}" / "model_source.ant").read_text()
    kept = prov / "versions" / f"v{res['version']}" / "model_source.ant"
    assert kept.is_file(), "a version with a model must have a source to roll back onto"
    assert kept.read_text().strip(), "never an empty file — that reads as 'authored nothing'"
    assert kept.read_text() == base_src, "the text did not change, so neither should the source"


def test_a_fork_patches_its_BASE_not_head(served):
    """With `base_version=0`, the stored source is v0's source plus THIS call's operations — so
    v1's change is absent from it. Under structured edits this is the strong property: the
    operations apply to the source, so `child.source == parent.source + ops` holds exactly.

    (Matching is on symbols, never on exact formatting: libantimony re-renders `(k1*2)*A` as
    `k1*2*A`, so asserting the literal trips over the very render gap the production failures
    came from.)"""
    mcp, prov = served
    _v0 = _call(
        mcp,
        "make_changes",
        {
            "justification": "v0",
            "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "(k1*2)*A"}],
        },
    )
    _v1 = _call(
        mcp,
        "make_changes",
        {
            "justification": "v1",
            "model_edits": [{"op": "set_rate_law", "id": "r2", "antimony": "(k2*2)*B"}],
        },
    )
    v2 = _call(
        mcp,
        "make_changes",
        {
            "justification": "fork from v0",
            "base_version": 0,
            "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "(k1*3)*A"}],
        },
    )
    assert v2["committed"] is True
    submitted = (prov / "versions" / f"v{v2['version']}" / "model_source.ant").read_text()
    assert "k1*3" in submitted  # this call's patch is in it
    assert "k2*2" not in submitted, (
        "the fork's submitted text carries v1's change — it was patched from HEAD, not v0"
    )


def test_a_fork_restores_the_search_space_not_just_the_model(served):
    """`base_version=N` used to roll back the MODEL only, so a fork silently inherited whatever
    bounds the abandoned branch had widened, and the agent had to be asked to re-send
    `parameters` from memory. Going back to a version now brings its search space with it."""
    mcp, prov = served
    v0 = _call(
        mcp,
        "make_changes",
        {
            "justification": "v0",
            "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "(k1*1.0)*A"}],
        },
    )
    # widen k1 far beyond v0's box, then fork back to v0
    _call(
        mcp,
        "make_changes",
        {
            "justification": "widen k1",
            "parameters": [{"parameter_id": "k1", "lower_bound": 1e-9, "upper_bound": 1e9}],
        },
    )
    import csv as _csv

    def bounds(version):
        with open(prov / "versions" / f"v{version}" / "parameters.tsv") as fh:
            for r in _csv.DictReader(fh, delimiter="\t"):
                if r["parameterId"] == "k1":
                    return float(r["lowerBound"]), float(r["upperBound"])
        return None

    v0_box = bounds(v0["version"])
    fork = _call(
        mcp,
        "make_changes",
        {
            "justification": "fork from v0",
            "base_version": v0["version"],
            "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "(k1*3.0)*A"}],
        },
    )
    assert fork.get("committed") is True, fork
    assert bounds(fork["version"]) == v0_box, (
        f"fork kept the widened box {bounds(fork['version'])} instead of v0's {v0_box}"
    )


def test_an_explicit_bound_in_the_fork_call_wins_over_the_restored_one(served):
    """A caller can fork a version AND change one bound in a single transaction; the bound they
    name is the one that survives.

    Implemented by EXCLUSION, not by overwrite: this call's `parameters` run first, against the
    pristine table, and the restore then skips the columns they set. That order
    is load-bearing — the widening backstops compare against the PRE-EDIT bound, so restoring
    first would hand them v0's box as the baseline and let a third widening past the guard."""
    mcp, prov = served
    v0 = _call(
        mcp,
        "make_changes",
        {
            "justification": "v0",
            "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "(k1*1.0)*A"}],
        },
    )
    fork = _call(
        mcp,
        "make_changes",
        {
            "justification": "fork and rebound",
            "base_version": v0["version"],
            "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "(k1*2.0)*A"}],
            "parameters": [{"parameter_id": "k1", "lower_bound": 0.02, "upper_bound": 20.0}],
        },
    )
    assert fork.get("committed") is True, fork
    import csv as _csv

    with open(prov / "versions" / f"v{fork['version']}" / "parameters.tsv") as fh:
        row = next(r for r in _csv.DictReader(fh, delimiter="\t") if r["parameterId"] == "k1")
    assert (float(row["lowerBound"]), float(row["upperBound"])) == (0.02, 20.0)


def test_a_refit_keeps_the_base_versions_columns_its_parameters_do_not_set(served):
    """REGRESSION (PEtab-MCP#202 review): naming a parameter withheld its WHOLE restored search
    space, so refitting v0 with new bounds kept HEAD's scale, and with a new scale kept HEAD's box."""
    import csv as _csv

    mcp, prov = served

    def k1(version):
        with open(prov / "versions" / f"v{version}" / "parameters.tsv") as fh:
            row = next(r for r in _csv.DictReader(fh, delimiter="\t") if r["parameterId"] == "k1")
        return row["parameterScale"], float(row["lowerBound"]), float(row["upperBound"])

    edit = [{"op": "set_rate_law", "id": "r1", "antimony": "(k1*1.0)*A"}]
    v0 = _call(mcp, "make_changes", {"justification": "v0", "model_edits": edit})["version"]
    head = {"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 10.0, "scale": "lin"}
    _call(mcp, "make_changes", {"justification": "HEAD's k1", "parameters": [head]})

    def refit(**entry):
        out = _call(
            mcp,
            "make_changes",
            {
                "justification": "refit v0",
                "base_version": v0,
                "parameters": [{"parameter_id": "k1", **entry}],
            },
        )
        assert out.get("committed") is True, out
        return k1(out["version"])

    assert refit(lower_bound=1e-3, upper_bound=1e3) == ("log10", 1e-3, 1e3)
    assert refit(scale="log") == ("log", 1e-4, 100.0)


def test_a_fork_resumes_from_its_BASE_optimum_not_the_abandoned_branchs(served):
    """`base_version=N` restores that version's nominal values, which is what makes a fork RESUME
    from version N rather than merely re-use its bounds.

    Why it matters more than the bounds do: `_run_fit` writes each fitted value back into
    `nominalValue`, and `warm_start_guesses` seeds start 0 (plus perturbed neighbours) from that
    point. Restore the model and the box but not the point, and the fork sets off from the optimum
    of the branch the agent just decided to abandon — the one place we know it does not want to
    look. Measured here through `objective_before`, the objective AT the starting point before the
    optimizer moves: for a fork whose edit is numerically inert, it must land on v0's own fitted
    objective, not the abandoned v1's.
    """
    mcp, _ = served
    v0 = _call(
        mcp,
        "make_changes",
        {
            "justification": "v0",
            "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "k1*A"}],
        },
    )
    v0_nllh = v0["neg_log_likelihood"]
    # A branch with a genuinely different optimum, so "restored" and "inherited" are separable.
    v1 = _call(
        mcp,
        "make_changes",
        {
            "justification": "a branch worth abandoning",
            "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "k1*A*A"}],
        },
    )
    assert v1["committed"] is True
    assert abs(v1["neg_log_likelihood"] - v0_nllh) > 1e-6, (
        "the two branches fit equally well, so this test cannot tell the points apart"
    )
    # Fork back to v0 with an edit that renders differently but computes the same value: the
    # restored point is therefore still v0's optimum under the fork's own structure.
    fork = _call(
        mcp,
        "make_changes",
        {
            "justification": "fork from v0",
            "base_version": v0["version"],
            "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "(k1 + 0.0)*A"}],
        },
    )
    assert fork.get("committed") is True, fork
    assert _start_objective(fork) == pytest.approx(v0_nllh, rel=1e-6), (
        f"the fork started from {(fork.get('improvement') or {}).get('objective_before')}, "
        f"not v0's optimum {v0_nllh} — "
        f"it inherited the abandoned branch's point (v1 fit to {v1['neg_log_likelihood']})"
    )


def test_the_starting_point_comes_back_even_for_a_parameter_the_fork_rebounds(served):
    """`skip` withholds the SEARCH SPACE the caller claimed, never the starting point.

    The caller cannot ask for a point — `parameters` rejects `nominalValue` outright — so treating
    a named id as "handled" would mean that retuning one bound inside a fork silently dropped that
    parameter back to the abandoned branch's value. Fork-and-rebound is exactly the case where the
    agent is steering hardest, so it is the worst place to lose the rollback."""
    mcp, prov = served
    v0 = _call(
        mcp,
        "make_changes",
        {
            "justification": "v0",
            "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "k1*A"}],
        },
    )
    _call(
        mcp,
        "make_changes",
        {
            "justification": "a branch worth abandoning",
            "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "k1*A*A"}],
        },
    )
    fork = _call(
        mcp,
        "make_changes",
        {
            "justification": "fork and rebound k1",
            "base_version": v0["version"],
            "model_edits": [{"op": "set_rate_law", "id": "r1", "antimony": "(k1 + 0.0)*A"}],
            "parameters": [{"parameter_id": "k1", "lower_bound": 1e-3, "upper_bound": 1e3}],
        },
    )
    assert fork.get("committed") is True, fork
    assert _start_objective(fork) == pytest.approx(v0["neg_log_likelihood"], rel=1e-6), (
        "naming k1 in `parameters` cost the fork k1's restored starting point"
    )
    # ...and the caller's own box is the one that survived.
    import csv as _csv

    with open(prov / "versions" / f"v{fork['version']}" / "parameters.tsv") as fh:
        row = next(r for r in _csv.DictReader(fh, delimiter="\t") if r["parameterId"] == "k1")
    assert (float(row["lowerBound"]), float(row["upperBound"])) == (1e-3, 1e3)
