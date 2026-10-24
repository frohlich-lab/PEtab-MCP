"""``verify_reduction`` (:mod:`petab_mcp.reduction_check`): a statements model, checked through the
server against the model its rows build with no reduction, at its fitted values."""

from __future__ import annotations

import asyncio
import json

import pytest
from conftest import YAML_PATH

from petab_mcp import reduction_check as RC
from petab_mcp.errors import ModelUpdateError
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


def test_a_split_monomer_agrees_with_its_unreduced_expansion(tmp_path):
    """The enzyme's two sites are put on apart, so it is tracked as two blocks and its rate
    factor is their product: checked the same way, against the expansion over its four forms.
    The kinase's rates are pinned so that the doubly modified enzyme, and so B, is not near 0."""
    mcp = _built(
        tmp_path,
        statements=[
            "conv: A -> B by E{pY1, pS2} | mass_action",
            "y1: K phosphorylates E at Y1 | indra_default kf=2",
            "s2: K phosphorylates E at S2 | indra_default kf=3",
        ],
    )
    out = _call(mcp, "verify_reduction", {})
    assert out["status"] == RC.STATUS_OK, out
    assert out["network"]["split"] == {"E": {"E_Y1": ["Y1"], "E_S2": ["S2"]}}
    assert abs(out["worst"]["unreduced"]) > 1e-3


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


def _problem(
    tmp_path,
    statements,
    readouts,
    *,
    rates,
    starts=(),
    experiments=(("e", "-inf", ""), ("e", 0, "")),
    conditions=(),
    times=(0.0, 10.0, 100.0),
):
    """A PEtab v2 problem with one observable per readout (id to agent JSON), measured in each
    experiment at ``times``, its ``rates`` shipped as fixed rows, built from ``statements`` and
    ``starts``. By default one experiment, measured from t=0 after a pre-equilibration."""
    problem = tmp_path / "problem"
    problem.mkdir()
    (problem / "model.ant").write_text("model preeq\n  species X; X = 1;\nend\n")
    (problem / "observables.tsv").write_text(
        "observableId\tobservableFormula\tnoiseFormula\n"
        + "".join(f"obs_{r}\t{r}\tsigma\n" for r in readouts)
    )
    (problem / "parameters.tsv").write_text(
        "parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n"
        "sigma\tlin\t0.001\t100\t1\tfalse\n"
        + "".join(f"{k}\tlog10\t1e-12\t1e6\t{v}\tfalse\n" for k, v in rates.items())
    )
    (problem / "experiments.tsv").write_text(
        "experimentId\ttime\tconditionId\n"
        + "".join("\t".join(map(str, e)) + "\n" for e in experiments)
    )
    files = ["experiment_files:\n- experiments.tsv\n"]
    if conditions:
        (problem / "conditions.tsv").write_text(
            "conditionId\ttargetId\ttargetValue\n"
            + "".join("\t".join(map(str, c)) + "\n" for c in conditions)
        )
        files.append("condition_files:\n- conditions.tsv\n")
    (problem / "measurements.tsv").write_text(
        "observableId\texperimentId\ttime\tmeasurement\n"
        + "".join(
            f"obs_{r}\t{e}\t{t}\t1\n"
            for e in dict.fromkeys(e for e, *_ in experiments)
            for t in times
            for r in readouts
        )
    )
    (problem / "problem.yaml").write_text(
        "format_version: 2.0.0\nparameter_files:\n- parameters.tsv\nmodel_files:\n  preeq:\n"
        "    location: model.ant\n    language: antimony\nmeasurement_files:\n- measurements.tsv\n"
        "observable_files:\n- observables.tsv\n" + "".join(files)
    )
    entities = tmp_path / "entities.json"
    entities.write_text(json.dumps(readouts))
    mcp = create_server(
        str(problem / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        build_root=str(tmp_path / "amici"),
        resolver=StubReferenceResolver(),
        auto_fit=False,
        entity_map=str(entities),
    )
    change = {"justification": "built from statements", "statements": list(statements)}
    out = _call(mcp, "make_changes", change | ({"starts": list(starts)} if starts else {}))
    assert out["applied"] is True, out
    return mcp


_pY1 = {"mod_type": "phosphorylation", "residue": "Y", "position": "1"}
_pS2 = {"mod_type": "phosphorylation", "residue": "S", "position": "2"}
_E_READOUTS = {"pE": {"name": "E", "mods": [_pY1]}, "ppE": {"name": "E", "mods": [_pY1, _pS2]}}


def _preequilibrated(tmp_path, rates=_RATES, e_amount=1000):
    """:data:`_SELF_PHOSPHORYLATION`, pre-equilibrated and measured from t=0, ``e_amount`` of E."""
    return _problem(
        tmp_path,
        _SELF_PHOSPHORYLATION,
        _E_READOUTS,
        rates=rates,
        starts=[{"agent": "E", "amount": e_amount}, {"agent": "P", "amount": 1}],
    )


def _scaled(monkeypatch, rate: str, factor: float) -> None:
    """Build the unreduced model with ``rate`` multiplied by ``factor``: a reduction that is wrong."""
    unreduced = RC._unreduced_antimony

    def wrong(session, record):
        text = unreduced(session, record)
        assert f"{rate}*" in text
        return text.replace(f"{rate}*", f"{rate}*{factor!r}*", 1)

    monkeypatch.setattr(RC, "_unreduced_antimony", wrong)


def test_a_slow_steady_state_is_no_mismatch(tmp_path, monkeypatch):
    """perica_egf's r2_v2 (run 61180684_2, verify-98): the two models were 1.3e-4 apart at t=0,
    every tolerance at 1e-10 but the steady state's. That is the pre-equilibration's own error:
    with its tolerances pinned tight, and a difference read against the unreduced model's own
    change between two tolerance levels, it is no mismatch; while a wrong rate still is."""
    mcp = _preequilibrated(tmp_path)
    out = _call(mcp, "verify_reduction", {})
    assert out["status"] == RC.STATUS_OK, out
    with monkeypatch.context() as patch:
        # The rule before: rtol 1e-10, AMICI's steady-state defaults, no spread allowed, and no
        # level refused for failing to reproduce itself at the next.
        old = RC.Tolerances(1e-10, 1e-12, 1e-8, 1e-16)
        patch.setattr(RC, "LEVELS", (old, RC.Tolerances(*(10 * t for t in old))))
        patch.setattr(RC, "SPREAD", 0)
        patch.setattr(RC, "CAP", float("inf"))
        patch.setattr(RC, "NLL_CAP", float("inf"))
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


@pytest.mark.slow
def test_a_slow_steady_state_hides_no_real_error(tmp_path, monkeypatch):
    """At rates of 1e-12 and 1e-10 the steady state is so slow that each model's own change from
    one tolerance level to the next is 1e-5 of the value. Ten times that, uncapped, let a rate
    wrong by 1% pass (the #116 review); capped at 1e-3 of the value, it is a mismatch."""
    rates = {"y1_kf": 1e-12, "s2_kf": 1e-12, "dy1_kf": 1e-10, "ds2_kf": 1e-10}
    mcp = _preequilibrated(tmp_path, rates)
    assert _call(mcp, "verify_reduction", {})["status"] == RC.STATUS_OK
    _scaled(monkeypatch, "dy1_kf", 1.01)
    assert _call(mcp, "verify_reduction", {})["status"] == RC.STATUS_MISMATCH


@pytest.mark.slow
def test_a_stiff_steady_state_is_decided_not_a_mismatch(tmp_path):
    """A million of E phosphorylating itself fast: at the tightest levels neither model reaches
    its steady state in time, and at one the reduced model's came back far off and reported
    success (786,256 against 998,002, which scipy confirms for both). Compared where both models
    reproduce themselves at the next level, they agree."""
    rates = {"y1_kf": 0.1, "s2_kf": 0.1, "dy1_kf": 100.0, "ds2_kf": 100.0}
    out = _call(_preequilibrated(tmp_path, rates, e_amount=1e6), "verify_reduction", {})
    assert out["status"] == RC.STATUS_OK and out["level"] > 0, out


@pytest.mark.slow
def test_two_experiments_one_of_them_fast_are_decided(tmp_path):
    """One pre-equilibration with a thousand of E, one with a million: the fast one moves the
    comparison to a looser level, and no longer makes the whole check undecided (the #116 review)."""
    rates = {"y1_kf": 0.1, "s2_kf": 0.1, "dy1_kf": 100.0, "ds2_kf": 100.0}
    mcp = _problem(
        tmp_path,
        _SELF_PHOSPHORYLATION,
        _E_READOUTS,
        rates=rates,
        starts=[{"agent": "E", "amount": "E_total"}, {"agent": "P", "amount": 1}],
        experiments=[
            ("lo", "-inf", "c_lo"),
            ("lo", 0, "c_lo"),
            ("hi", "-inf", "c_hi"),
            ("hi", 0, "c_hi"),
        ],
        conditions=[("c_lo", "E_total", 1e3), ("c_hi", "E_total", 1e6)],
        times=(0.0, 10.0),
    )
    assert _call(mcp, "verify_reduction", {})["status"] == RC.STATUS_OK


@pytest.mark.slow
def test_a_doubled_rate_on_a_tiny_readout_is_a_mismatch(tmp_path, monkeypatch):
    """A homodimer kinase counted twice, the very error the species-mode catalyst prevents, at
    seed rates that keep the readout near 1e-13: a fixed absolute floor of 1e-11 let it pass (the
    #116 review); the compared level's own absolute tolerance does not."""
    mcp = _problem(
        tmp_path,
        [
            "dim: E binds E | indra_default",
            "p: E{!E} phosphorylates S at S2 | indra_default",
            "dp: P dephosphorylates S at S2 | indra_default",
        ],
        {"pS": {"name": "S", "mods": [_pS2]}},
        rates={},
        experiments=[("e", 0, "")],
        times=(0.0, 1.0, 5.0, 20.0),
    )
    assert _call(mcp, "verify_reduction", {})["status"] == RC.STATUS_OK
    _scaled(monkeypatch, "p_kf", 2.0)
    assert _call(mcp, "verify_reduction", {})["status"] == RC.STATUS_MISMATCH


@pytest.mark.parametrize(
    "failure",
    [RuntimeError("simulated AMICI compile failure"), ModelUpdateError("simulated refused build")],
)
def test_a_failure_of_the_check_is_written_down_as_an_error(tmp_path, monkeypatch, failure):
    """A crash, or a refusal, building the unreduced side: no status and no file, before (the
    #116 review); now ``error``, with why, written beside the version like any result."""
    mcp = _built(tmp_path, statements=[_CATALYSED])

    def fail(session, antimony):
        raise failure

    monkeypatch.setattr(RC, "_reconciled", fail)
    out = _call(mcp, "verify_reduction", {})
    assert out["status"] == RC.STATUS_ERROR and type(failure).__name__ in out["reason"]
    stored = json.loads(
        (tmp_path / "prov" / "versions" / "v0" / "reduction_check.json").read_text()
    )
    assert stored["status"] == RC.STATUS_ERROR
