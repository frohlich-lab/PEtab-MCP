"""Zero-shotting the model with ``init_model`` (only when none is initialized)."""

from __future__ import annotations

import shutil
from pathlib import Path

import libsbml
import pytest

from petab_mcp import operations as O
from petab_mcp import parameters as P
from petab_mcp.operations import OperationError
from petab_mcp.references import StubReferenceResolver
from petab_mcp.session import PetabSession

DATA_DIR = Path(__file__).parent / "data" / "toy_v1"

# A correct reversible conversion model authored from scratch in Antimony.
REVERSIBLE_ANTIMONY = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.25;
  k2 = 0.05;
  r1: A => B; k1 * A;
  r2: B => A; k2 * B;
end
"""


def _write_empty_sbml(path: Path, model_id: str = "conversion_reaction") -> None:
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId(model_id)
    model.setMetaId(model_id)
    libsbml.writeSBMLToFile(doc, str(path))


@pytest.fixture
def empty_yaml(tmp_path) -> str:
    """A PEtab problem that ships WITHOUT a model (empty SBML, no model.ant)."""
    work = tmp_path / "empty"
    work.mkdir()
    for name in ("problem.yaml", "conditions.tsv", "measurements.tsv",
                 "observables.tsv", "parameters.tsv"):
        shutil.copy(DATA_DIR / name, work / name)
    _write_empty_sbml(work / "conversion_reaction.xml")
    return str(work / "problem.yaml")


@pytest.fixture
def empty_session(empty_yaml, tmp_path) -> PetabSession:
    return PetabSession(
        empty_yaml,
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )


def _species(session):
    sm = session.model.sbml_model
    return {sm.getSpecies(i).getId() for i in range(sm.getNumSpecies())}


def _reactions(session):
    sm = session.model.sbml_model
    return {sm.getReaction(i).getId() for i in range(sm.getNumReactions())}


def _ids(session):
    return {r["parameterId"] for r in P.parameters_to_records(session.problem)}


# -- fresh problem: no model initialized -----------------------------------
def test_empty_problem_has_no_model(empty_session):
    assert empty_session.has_model() is False
    # the initial version is still snapshotted (an empty model)
    assert empty_session.provenance.current_version == 0


def test_init_model_zero_shots(empty_session):
    res = O.init_model(
        empty_session, REVERSIBLE_ANTIMONY,
        justification="reversible conversion A<->B", references=["PMID:1"],
    )
    assert res["committed"] is True
    assert res["operation"] == "init_model"
    assert empty_session.has_model() is True
    assert _species(empty_session) == {"A", "B"}
    assert _reactions(empty_session) == {"r1", "r2"}
    # rate constants + parametric-IC params synced into the table; noise param kept
    assert {"k1", "k2", "A_init", "B_init", "sigma_B"} <= _ids(empty_session)


def test_init_model_references_optional(empty_session):
    # references are optional for model operations (like add_reaction et al.)
    res = O.init_model(empty_session, REVERSIBLE_ANTIMONY, justification="reversible A<->B")
    assert res["committed"] is True
    assert res["references"] == []
    assert empty_session.has_model() is True


def test_init_model_uses_antimony_header_model_id(empty_session):
    # the Antimony header names the model, not the empty-load placeholder id
    O.init_model(empty_session, REVERSIBLE_ANTIMONY, justification="build")
    assert empty_session.model.model_id == "conversion_reaction"


def test_init_model_requires_justification(empty_session):
    with pytest.raises(OperationError):
        O.init_model(empty_session, REVERSIBLE_ANTIMONY, justification="  ", references=["PMID:1"])


def test_init_model_requires_nonempty_antimony(empty_session):
    with pytest.raises(OperationError):
        O.init_model(empty_session, "   ", justification="x", references=["PMID:1"])


def test_init_model_rejects_unparseable_antimony(empty_session):
    from petab_mcp.errors import ModelUpdateError

    with pytest.raises(ModelUpdateError):
        O.init_model(empty_session, "this is not antimony {{{", justification="x",
                     references=["PMID:1"])
    # a failed init leaves the problem model-less (atomic revert)
    assert empty_session.has_model() is False


def test_incremental_edits_refuse_without_a_model(empty_session):
    with pytest.raises(OperationError):
        O.add_reaction(empty_session, "r1: A => B; k1*A", justification="x", references=["PMID:1"])
    with pytest.raises(OperationError):
        O.change_rate_law(empty_session, "r1", "k1*A", justification="x", references=["PMID:1"])
    with pytest.raises(OperationError):
        O.remove_reaction(empty_session, "r1", justification="x", references=["PMID:1"])


def test_can_edit_after_init(empty_session):
    O.init_model(empty_session, REVERSIBLE_ANTIMONY, justification="build", references=["PMID:1"])
    res = O.add_reaction(empty_session, "rdeg: A => ; kdeg*A",
                         justification="degradation", references=["PMID:2"])
    assert res["committed"] is True
    assert "rdeg" in _reactions(empty_session)


# -- existing model: init_model is refused ---------------------------------
def test_init_model_refused_when_model_exists(session):
    assert session.has_model() is True
    with pytest.raises(OperationError):
        O.init_model(session, REVERSIBLE_ANTIMONY, justification="x", references=["PMID:1"])
