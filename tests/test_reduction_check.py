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


# -- a pre-equilibrated model whose steady state is slow to settle -----------------------------
#: E phosphorylates itself at Y1 and, so phosphorylated, at S2; P takes both off. Every rate is
#: slow, so the steady state the pre-equilibration accepts is off by its tolerance over those rates.
_SELF_PHOSPHORYLATION = [
    "y1: E phosphorylates E at Y1 | indra_default",
    "s2: E{pY1} phosphorylates E at S2 | indra_default",
    "dy1: P dephosphorylates E at Y1 | indra_default",
    "ds2: P dephosphorylates E at S2 | indra_default",
]
_RATES = {"y1_kf": 1e-9, "s2_kf": 1e-9, "dy1_kf": 1e-7, "ds2_kf": 1e-7}


def _preequilibrated(tmp_path):
    """A PEtab v2 problem measured at t=0 after a pre-equilibration and later, its rates shipped
    as fixed rows, built from :data:`_SELF_PHOSPHORYLATION` with 1,000 of E."""
    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "model.ant").write_text("model preeq\n  species E; E = 1;\nend\n")
    (problem / "observables.tsv").write_text(
        "observableId\tobservableFormula\tnoiseFormula\nobs_pE\tpE\tsigma\nobs_ppE\tppE\tsigma\n"
    )
    (problem / "parameters.tsv").write_text(
        "parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n"
        "sigma\tlin\t0.001\t100\t1\tfalse\n"
        + "".join(f"{k}\tlog10\t1e-12\t1e6\t{v}\tfalse\n" for k, v in _RATES.items())
    )
    (problem / "experiments.tsv").write_text("experimentId\ttime\tconditionId\ne\t-inf\t\ne\t0\t\n")
    (problem / "measurements.tsv").write_text(
        "observableId\texperimentId\ttime\tmeasurement\n"
        + "".join(f"{o}\te\t{t}\t1\n" for t in (0.0, 10.0, 100.0) for o in ("obs_pE", "obs_ppE"))
    )
    (problem / "problem.yaml").write_text(
        "format_version: 2.0.0\nparameter_files:\n- parameters.tsv\nmodel_files:\n  preeq:\n"
        "    location: model.ant\n    language: antimony\nmeasurement_files:\n- measurements.tsv\n"
        "observable_files:\n- observables.tsv\nexperiment_files:\n- experiments.tsv\n"
    )
    pY1 = {"mod_type": "phosphorylation", "residue": "Y", "position": "1"}
    pS2 = {"mod_type": "phosphorylation", "residue": "S", "position": "2"}
    entities = tmp_path / "entities.json"
    entities.write_text(
        json.dumps({"pE": {"name": "E", "mods": [pY1]}, "ppE": {"name": "E", "mods": [pY1, pS2]}})
    )
    mcp = create_server(
        str(problem / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        build_root=str(tmp_path / "amici"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
        entity_map=str(entities),
    )
    out = _call(
        mcp,
        "make_changes",
        {
            "justification": "E phosphorylates itself",
            "statements": _SELF_PHOSPHORYLATION,
            "starts": [{"agent": "E", "amount": 1000}, {"agent": "P", "amount": 1}],
        },
    )
    assert out["applied"] is True, out
    return mcp


def test_a_slow_steady_state_is_no_mismatch(tmp_path, monkeypatch):
    """perica_egf's r2_v2 (run 61180684_2, verify-98): the two models were 1.3e-4 apart at t=0,
    every tolerance at 1e-10 but the steady state's. That is the pre-equilibration's own error:
    with its tolerances pinned tight, and a difference read against the unreduced model's own
    change between two tolerance levels, it is no mismatch; while a wrong rate still is."""
    mcp = _preequilibrated(tmp_path)
    out = _call(mcp, "verify_reduction", {})
    assert out["status"] == RC.STATUS_OK, out
    with monkeypatch.context() as patch:
        # The rule before: rtol 1e-10, AMICI's steady-state defaults, and no spread allowed.
        patch.setattr(RC, "TIGHT", RC.Tolerances(1e-10, 1e-12, 1e-8, 1e-16))
        patch.setattr(RC, "SPREAD", 0)
        before = _call(mcp, "verify_reduction", {})
    assert before["status"] == RC.STATUS_MISMATCH and before["worst"]["time"] == 0.0, before
    unreduced = RC._unreduced_antimony
    monkeypatch.setattr(
        RC,
        "_unreduced_antimony",
        lambda session, record: unreduced(session, record).replace("dy1_kf*", "dy1_kf*0.5*", 1),
    )
    wrong = _call(mcp, "verify_reduction", {})
    assert wrong["status"] == RC.STATUS_MISMATCH and wrong["margin"] > 1, wrong
