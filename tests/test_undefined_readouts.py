"""A readout the observable table names but the model never defines: the commit-time remedy
(``validation._undefined_readout_issues``), the seed model's handling of readouts
(``reconcile.seed_model_antimony``), how the tools show one before the first build, and the
refusal of a readout that reads itself.

The harness blinds observables to ``readout_*`` symbols, so a model that forgets to define one
must be refused with the remedy and the biology attached, and a seed must declare a readout as a
parameter rather than a species. The blinding itself is tested in gemot.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from petab_mcp import model as model_mod
from petab_mcp import operations as O
from petab_mcp import reconcile as R
from petab_mcp.operations import OperationError
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server
from petab_mcp.session import PetabSession
from petab_mcp.validation import _undefined_readout_issues


def _prob(formulas: dict[str, str], noise: dict[str, str] | None = None):
    """A minimal problem-shaped object carrying just an observable_df."""
    odf = pd.DataFrame(
        {
            "observableFormula": pd.Series(formulas),
            "noiseFormula": pd.Series(noise or {k: f"noiseParameter1_{k}" for k in formulas}),
        }
    )
    odf.index.name = "observableId"

    class _P:
        observable_df = odf

    return _P()


def _sbml_with(
    ids: list[str],
    with_reaction: bool = True,
    rules: list[str] | None = None,
    rate_rules: list[str] | None = None,
):
    """Returns (document, model). The DOCUMENT must be kept alive by the caller: libsbml's
    SWIG proxies do not own their C++ objects, so dropping the document frees the model out
    from under the proxy and the next `getElementBySId` segfaults the interpreter."""
    import libsbml

    doc = libsbml.SBMLDocument(3, 1)
    m = doc.createModel()
    for sid in ids:
        p = m.createParameter()
        p.setId(sid)
        p.setConstant(True)
        p.setValue(1.0)
    for sid in rules or []:
        r = m.createAssignmentRule()
        r.setVariable(sid)
        r.setFormula("1")
    for sid in rate_rules or []:
        r = m.createRateRule()
        r.setVariable(sid)
        r.setFormula("1")
    if with_reaction:
        # `_undefined_readout_issues` self-skips a model with no dynamics (the v0 seed), so a
        # test of the CHECK needs one reaction to get past that.
        rx = m.createReaction()
        rx.setId("r1")
        rx.setReversible(False)
        rx.setFast(False)
    return doc, m


def _lintable(
    formulas: dict[str, str],
    model_ids: list[str],
    names: dict | None = None,
    with_reaction: bool = True,
    rules: list[str] | None = None,
    rate_rules: list[str] | None = None,
):
    prob = _prob(formulas)
    prob.observable_df["observableName"] = pd.Series(names or {})
    doc, sm = _sbml_with(model_ids, with_reaction=with_reaction, rules=rules, rate_rules=rate_rules)

    class _M:
        _doc = doc  # lifetime anchor — see _sbml_with
        sbml_model = sm

    prob.model = _M()
    prob.parameter_df = pd.DataFrame(index=pd.Index([], name="parameterId"))
    return prob


def test_an_undefined_readout_is_an_error_carrying_remedy_and_biology():
    prob = _lintable(
        {"a": "scale_a * readout_a"}, model_ids=["scale_a"], names={"a": "immunoblot signal for X"}
    )
    issues = _undefined_readout_issues(prob)
    assert len(issues) == 1 and issues[0]["level"] == "error"
    msg = issues[0]["message"]
    assert "readout_a" in msg and "set_assignment" in msg
    assert "It measures: immunoblot signal for X." in msg


def test_a_defined_readout_passes():
    """Defined means an ASSIGNMENT RULE — that is what a measurement mapping is."""
    prob = _lintable({"a": "scale_a * readout_a"}, model_ids=["scale_a"], rules=["readout_a"])
    assert _undefined_readout_issues(prob) == []


def test_non_readout_symbols_are_not_this_checks_business():
    """A missing ordinary species is petab lint's job — one generic error, not two."""
    prob = _lintable({"a": "some_species"}, model_ids=[])
    assert _undefined_readout_issues(prob) == []


def test_a_readout_declared_as_a_SPECIES_is_still_undefined():
    """The regression that made this whole guard a no-op. The seed declares one species per
    observable-formula identifier so v0 lints — which includes every `readout_*` — so "the id
    resolves to an SBML element" is true by construction. A model leaving each readout a free
    constant would then commit: on blasi that is 16 constants against 16 steady-state observable
    means, fitting them exactly while skipping the measurement-mapping task entirely."""
    prob = _lintable({"a": "readout_a"}, model_ids=["readout_a"], with_reaction=True)
    issues = _undefined_readout_issues(prob)
    assert len(issues) == 1, "a readout declared as a species must NOT count as defined"
    assert "ASSIGNMENT RULE" in issues[0]["message"]


def test_the_dynamics_free_seed_is_not_flagged():
    """v0 has no reactions and nothing to map a readout onto — the same self-skip
    `_unused_condition_target_issues` uses. Flagging it would make the seed itself invalid."""
    prob = _lintable({"a": "readout_a"}, model_ids=["readout_a"], with_reaction=False)
    assert _undefined_readout_issues(prob) == []


def test_a_missing_description_does_not_render_as_nan():
    """`or ""` does not catch a NaN cell — float('nan') is truthy — so the agent was told
    "It measures: nan." """
    import math

    prob = _lintable({"a": "readout_a"}, model_ids=[], with_reaction=True, names={"a": math.nan})
    msg = _undefined_readout_issues(prob)[0]["message"]
    assert "nan" not in msg.lower(), msg


def test_the_seed_declares_a_readout_as_a_parameter_not_a_species():
    """The trap the first blinded sweep measured: the seed declared `readout_*` as SPECIES, agents
    copied that into their own models, wired the readout into reactions as a state, and then could
    not assign it — 4 of the sweep's 27 rejections were exactly this over-determination refusal,
    all on fujita (the example with non-identity mappings). A constant parameter lints and
    simulates identically at v0 and suggests nothing about being a state. The seed must also stay
    DYNAMICS-FREE, so a `:=` rule (which would flip `has_dynamics` and with it `model_authored`)
    is not an option."""
    from petab_mcp.reconcile import seed_model_antimony

    odf = pd.DataFrame(
        {"observableFormula": pd.Series({"a": "scale_a * readout_a", "b": "some_species"})}
    )
    odf.index.name = "observableId"

    class _P:
        observable_df = odf
        condition_df = None
        parameter_df = pd.DataFrame(index=pd.Index(["scale_a"], name="parameterId"))

    text = seed_model_antimony(_P())
    assert "species some_species;" in text
    assert "readout_a = 0.1;" in text
    assert "species readout_a" not in text, (
        "a seeded readout species is the over-determination trap"
    )
    assert ":=" not in text, "a seeded rule would make the dynamics-free seed count as authored"


def test_a_pure_readout_table_demotes_every_readout():
    """No readout is ever a species, even when the formulas name nothing else.

    Two earlier attempts tied this to the AMICI zero-species segfault: keeping ALL readouts as
    species (which left boehm, blasi, crauste, perelson and rahman with the trap the demotion
    exists to remove), then keeping exactly one. `reconcile.ensure_one_species` carries that
    invariant with a placeholder instead, so the seed is free to say what is true — a measurement
    mapping is not a state."""
    from petab_mcp.reconcile import seed_model_antimony

    odf = pd.DataFrame({"observableFormula": pd.Series({"a": "readout_a", "b": "readout_b"})})
    odf.index.name = "observableId"

    class _P:
        observable_df = odf
        condition_df = None
        parameter_df = pd.DataFrame(index=pd.Index([], name="parameterId"))

    text = seed_model_antimony(_P())
    assert "species readout_" not in text, text
    assert "readout_a = 0.1;" in text and "readout_b = 0.1;" in text


def test_a_rate_rule_model_with_no_reactions_is_still_checked():
    """The self-skip must be `has_dynamics`, not a reaction count.

    ODE-style Antimony authoring produces a model that is fully dynamic with ZERO reactions, so
    a bare `getNumReactions() == 0` skipped exactly those models -- letting one commit with every
    readout left as a free constant, which is the loophole this check exists to close.
    """
    prob = _lintable(
        {"a": "readout_a"}, model_ids=["readout_a", "x"], with_reaction=False, rate_rules=["x"]
    )
    issues = _undefined_readout_issues(prob)
    assert len(issues) == 1 and "readout_a" in str(issues[0])


def test_a_rate_rule_model_that_defines_its_readout_passes():
    prob = _lintable(
        {"a": "readout_a"},
        model_ids=["readout_a", "x"],
        with_reaction=False,
        rate_rules=["x"],
        rules=["readout_a"],
    )
    assert _undefined_readout_issues(prob) == []


# -- a blinded problem shipped without a model -----------------------------------
_MODEL = """model blinded
  compartment c = 1;
  species A in c, B in c;
  A = 1;
  B = 0;
  k1 = 0.5;
  r1: A => B; k1 * A;
  {rule}
end
"""


@pytest.fixture
def blinded(empty_yaml):
    """The toy as a blinded benchmark ships: no model, its observable reading ``readout_B``."""
    observables = Path(empty_yaml).parent / "observables.tsv"
    observables.write_text(observables.read_text().replace("\tB\t", "\treadout_B\t"))
    return empty_yaml


def _call(mcp, name, args=None):
    return asyncio.run(mcp.call_tool(name, args or {})).structured_content


def _server(yaml_path, tmp_path):
    return create_server(
        yaml_path, provenance_dir=str(tmp_path / "prov"), resolver=StubReferenceResolver()
    )


def test_before_the_first_build_a_readout_is_a_mapping_to_define(blinded, tmp_path):
    """The seed's constant `readout_B` got an estimated row, so every listing called it a free
    `observable_scaling` parameter and agents scaled it by itself (`readout_x := readout_x * X`)."""
    mcp = _server(blinded, tmp_path)
    described = _call(mcp, "describe_problem")
    assert described["free_parameters"] == []
    assert described["readouts_to_define"] == ["readout_B"]
    rows = _call(mcp, "get_parameters")["result"]
    assert "readout_B" not in {r["parameterId"] for r in rows}
    structure = _call(mcp, "get_model_structure")
    assert structure["readouts"] == {"readout_B": None}
    assert "readout_B" not in structure["parameters"]
    model = _call(mcp, "get_model")["result"]
    assert 'readout_B is "measurement mapping, which your model defines' in model


def test_a_defined_readout_is_shown_by_its_rule_and_never_gets_a_row(blinded, tmp_path):
    session = PetabSession(
        blinded, provenance_dir=str(tmp_path / "prov"), resolver=StubReferenceResolver()
    )
    assert "readout_B" not in R.classify_inner_parameters(session.problem)
    res = O.set_model(session, _MODEL.format(rule="readout_B := 2 * B;"), justification="B")
    assert res["applied"] is True
    assert "readout_B" not in session.problem.parameter_df.index
    assert session.describe()["readouts_to_define"] == []
    assert model_mod.model_structure(session)["readouts"] == {"readout_B": "2 * B"}


@pytest.mark.parametrize(
    "route",
    [
        {"antimony": _MODEL.format(rule="readout_B := readout_B * B;")},
        {"model_edits": [{"op": "set_assignment", "id": "readout_B", "antimony": "readout_B*B"}]},
    ],
    ids=["antimony", "set_assignment"],
)
def test_a_readout_that_reads_itself_is_refused_with_what_a_readout_is(blinded, tmp_path, route):
    """libantimony only reports a loop, which says nothing about what the readout is."""
    mcp = _server(blinded, tmp_path)
    res = _call(mcp, "make_changes", {**route, "justification": "scale the readout"})
    assert res["applied"] is False
    assert "'readout_B' is not a parameter to scale" in res["error"]
    assert _call(mcp, "describe_problem")["model_authored"] is False


def test_any_symbol_that_assigns_itself_is_refused(session):
    with pytest.raises(OperationError, match="'x' reads the symbol it assigns") as exc:
        O.set_model(session, _MODEL.format(rule="x := 2 * x + B;"), justification="loop")
    assert exc.value.cause == "self_assignment" and "not a parameter to scale" not in str(exc.value)
