from __future__ import annotations

import re
import shutil
from pathlib import Path

import libsbml
import pytest

from petab_mcp import protocols as Pr
from petab_mcp.errors import ProtocolError
from petab_mcp.problem import WorkingProblem
from petab_mcp.references import StubReferenceResolver
from petab_mcp.session import PetabSession

_RAMP = "piecewise(EGF_rate*time + EGF_0, time <= EGF_end, 0)"
_BASE_INPUTS = "inputId\tformula\tconditionInputs\n{rows}\n"


def _problem(tmp_path, rows: str, targets=("EGF_0", "EGF_rate", "EGF_end")) -> Path:
    """The toy problem, one condition setting each of ``targets`` to 1 in every measurement's
    experiment, and ``rows`` as its ``inputs.tsv``."""
    work = tmp_path / "problem"
    shutil.copytree(Path(__file__).parent / "data" / "toy_v2", work)
    (work / "conditions.tsv").write_text(
        "conditionId\ttargetId\ttargetValue\n" + "".join(f"c0\t{t}\t1\n" for t in targets)
    )
    (work / "experiments.tsv").write_text("experimentId\ttime\tconditionId\ne0\t0\tc0\n")
    meas = work / "measurements.tsv"
    meas.write_text(meas.read_text().replace("obs_B\t\t", "obs_B\te0\t"))
    yaml = work / "problem.yaml"
    yaml.write_text(
        yaml.read_text()
        + "condition_files:\n- conditions.tsv\nexperiment_files:\n- experiments.tsv\n"
    )
    (work / Pr.FILENAME).write_text(_BASE_INPUTS.format(rows=rows))
    return yaml


def test_fujitas_ramp_is_evaluated_and_shown_to_the_antimony_route(tmp_path):
    """The protocol reached the Antimony agent only as prose in PEtabAgentBench's task text, a
    third copy beside inputs.tsv and the entity map; the server now serves it from the file."""
    yaml = _problem(tmp_path, f"EGF\t{_RAMP}\tEGF_0;EGF_rate;EGF_end")
    session = PetabSession(yaml, provenance_dir=tmp_path / "prov", resolver=StubReferenceResolver())
    (egf,) = session.inputs
    values = {"EGF_0": 1.0, "EGF_rate": 2.0, "EGF_end": 1.5}
    assert egf.evaluate(values, [0.0, 1.0, 2.0]).tolist() == [1.0, 3.0, 0.0]
    assert session.describe()["inputs"] == [{"input": "EGF", "rule": f"EGF := {_RAMP}"}]


def test_no_file_declares_no_inputs(tmp_path):
    yaml = _problem(tmp_path, "")
    (yaml.parent / Pr.FILENAME).unlink()
    assert Pr.read(yaml.parent) == ()


def test_an_input_names_only_condition_targets_and_time(tmp_path):
    """Three PEtabAgentBench#6 entity maps put fitted constants in an input (Boehm's Epo
    degradation, Fiedler's stimulus time constants, Oliveira's transmission steps), handing the
    agent part of the reference's model where it can neither see nor change it."""
    yaml = _problem(tmp_path, "S\tdose*exp(-k1*time)\tdose;k1", targets=("dose",))
    with pytest.raises(ProtocolError, match=re.escape("'S' names k1 (estimated)")):
        Pr.load(yaml.parent, WorkingProblem.from_yaml(yaml))


@pytest.mark.parametrize(
    ("row", "refusal"),
    [
        (f"EGF\t{_RAMP}\tEGF_0;EGF_rate", "but the formula reads EGF_0;EGF_end;EGF_rate"),
        ("EGF\tpiecewise(EGF_0, time <\tEGF_0", "is no PEtab math"),
        ("INS\tEGF_0\tEGF_0", "does not change within a period; set it in the condition table"),
        (f"EGF\t{_RAMP}\tEGF_0;EGF_rate;EGF_end\nEGF\t{_RAMP}\tEGF_0;EGF_rate;EGF_end", "twice"),
    ],
)
def test_a_row_is_read_as_written_or_refused(tmp_path, row, refusal):
    """conditionInputs restates the formula's targets, so a stale list must not pass as another
    protocol's; and an input a period expresses is the condition table's, not a second declaration
    of it here."""
    with pytest.raises(ProtocolError, match=re.escape(refusal)):
        Pr.read(_problem(tmp_path, row).parent)


def _held(item: Pr.Input) -> libsbml.SBMLDocument:
    """A model of one species, held to ``item``; the document owns the model, so it is returned."""
    document = libsbml.SBMLDocument(3, 2)
    species = document.createModel().createSpecies()
    species.setId(item.id)
    species.setInitialAmount(0.0)
    Pr.hold(document.getModel(), item.id, item)
    return document


def test_a_held_input_follows_its_rule_in_petab_math():
    """libSBML's default parser reads `log` as log10, where PEtab's `log` is the natural log."""
    document = _held(Pr.Input("S", "log(dose)*time", ("dose",)))
    model = document.getModel()
    rule = model.getAssignmentRuleByVariable("S")
    assert libsbml.formulaToL3String(rule.getMath()) == "ln(dose) * time"
    assert model.getParameter("dose") is not None
    assert model.getSpecies("S").getBoundaryCondition()
