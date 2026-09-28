"""Authoring the whole model with ``set_model`` (builds the first model AND re-authors)."""

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

# A one-reaction re-author keeping only r2/k2 (drops r1 and its rate constant k1). Used to
# strand k1: reconciling this out removes k1's row, so bringing it back re-adds it as ESTIMATED.
ONLY_R2_ANTIMONY = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k2 = 0.1;
  r2: B => A; k2 * B;
end
"""

# A one-reaction re-author (drops r2 and its rate constant k2 vs REVERSIBLE_ANTIMONY).
ONE_REACTION_ANTIMONY = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.25;
  r1: A => B; k1 * A;
end
"""

# A genuinely-different re-author that KEEPS both reactions (and k1) and adds a B degradation —
# a real structural change (so it is not de-duplicated as a no-op) that still declares k1.
REVERSIBLE_PLUS_DEG_ANTIMONY = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.5;
  k2 = 0.1;
  kdeg = 0.01;
  r1: A => B; k1 * A;
  r2: B => A; k2 * B;
  rdeg: B => ; kdeg * B;
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
    for name in (
        "problem.yaml",
        "conditions.tsv",
        "measurements.tsv",
        "observables.tsv",
        "parameters.tsv",
    ):
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
def test_empty_problem_has_no_authored_model(empty_session):
    """A problem that ships model-less is SEEDED with a content-free model (see
    test_seed_model.py), so there is something to simulate from the start — but nothing has been
    authored, which is the flag everything downstream keys on."""
    assert empty_session.model_authored() is False
    assert empty_session.has_model() is True
    # the initial version is still snapshotted
    assert empty_session.provenance.current_version == -1


def test_set_model_builds_the_first_model(empty_session):
    res = O.set_model(
        empty_session,
        REVERSIBLE_ANTIMONY,
        justification="reversible conversion A<->B",
        references=["PMID:1"],
    )
    assert res["applied"] is True
    assert res["operation"] == "make_changes"
    assert empty_session.has_model() is True
    assert _species(empty_session) == {"A", "B"}
    assert _reactions(empty_session) == {"r1", "r2"}
    # rate constants + parametric-IC params synced into the table; noise param kept
    assert {"k1", "k2", "A_init", "B_init", "sigma_B"} <= _ids(empty_session)


def test_set_model_references_optional(empty_session):
    # references are optional for model operations (set_model / the parameter edits)
    res = O.set_model(empty_session, REVERSIBLE_ANTIMONY, justification="reversible A<->B")
    assert res["applied"] is True
    assert res["references"] == []
    assert empty_session.has_model() is True


def test_set_model_uses_antimony_header_model_id(empty_session):
    # the Antimony header names the model, not the empty-load placeholder id
    O.set_model(empty_session, REVERSIBLE_ANTIMONY, justification="build")
    assert empty_session.model.model_id == "conversion_reaction"


def test_set_model_requires_justification(empty_session):
    with pytest.raises(OperationError):
        O.set_model(empty_session, REVERSIBLE_ANTIMONY, justification="  ", references=["PMID:1"])


def test_set_model_requires_nonempty_antimony(empty_session):
    with pytest.raises(OperationError):
        O.set_model(empty_session, "   ", justification="x", references=["PMID:1"])


def test_set_model_rejects_unparseable_antimony(empty_session):
    from petab_mcp.errors import ModelUpdateError

    with pytest.raises(ModelUpdateError):
        O.set_model(
            empty_session, "this is not antimony {{{", justification="x", references=["PMID:1"]
        )
    # a failed author leaves the problem model-less (atomic revert)
    assert empty_session.model_authored() is False


# A model that (incorrectly) writes a declared parameter as a product stoichiometry coefficient.
# Antimony would freeze N_burst to its nominal and drop it, so set_model rejects it up front (#89).
PARAMETRIC_STOICH_ANTIMONY = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = A_init;
  B = B_init;
  A_init = 1.0;
  B_init = 0.0;
  k1 = 0.5;
  N_burst = 3;
  r1: A -> N_burst B; k1 * A;
end
"""


def test_set_model_rejects_parametric_stoichiometry(empty_session):
    from petab_mcp.errors import ValidationError

    # `N_burst` used as a product coefficient would be silently baked to a constant and dropped;
    # set_model rejects it before building anything, naming the parameter.
    with pytest.raises(ValidationError) as exc:
        O.set_model(empty_session, PARAMETRIC_STOICH_ANTIMONY, justification="burst amplification")
    assert any("N_burst" in issue for issue in (exc.value.issues or [])), exc.value.issues
    # Rejected up front — the problem is left model-less (nothing was built or committed).
    assert empty_session.model_authored() is False


def test_set_model_rejects_same_symbol_stoichiometry_on_both_sides(empty_session):
    from petab_mcp.errors import ValidationError

    # `n` as a coefficient on BOTH sides collides two SpeciesReference ids and fails the Antimony
    # conversion; set_model recognizes it as the same footgun and gives actionable guidance rather
    # than the opaque libSBML id-conflict error.
    both_sides = (
        "model m\n  species A; species B; A = 1; B = 0; k = 0.1; n = 2;\n"
        "  r1: n A -> n B; k * A;\nend"
    )
    with pytest.raises(ValidationError) as exc:
        O.set_model(empty_session, both_sides, justification="cooperative conversion")
    assert any("stoichiometry" in issue.lower() for issue in (exc.value.issues or [])), (
        exc.value.issues
    )
    assert empty_session.model_authored() is False


def test_can_reauthor_after_set_model(empty_session):
    # After the initial build, a structural change is another whole-model re-author that adds a
    # reaction.
    O.set_model(empty_session, REVERSIBLE_ANTIMONY, justification="build", references=["PMID:1"])
    res = O.set_model(
        empty_session,
        REVERSIBLE_PLUS_DEG_ANTIMONY,
        justification="add degradation",
        references=["PMID:2"],
    )
    assert res["applied"] is True
    assert "rdeg" in _reactions(empty_session)


# -- existing model: set_model re-authors it wholesale (never refused) ------
def test_set_model_replaces_existing_model(session):
    # THE point of set_model: NOT refused when a model already exists — the new Antimony
    # replaces the current model wholesale.
    assert session.has_model() is True
    res = O.set_model(session, ONE_REACTION_ANTIMONY, justification="drop the reverse reaction")
    assert res["applied"] is True
    assert res["operation"] == "make_changes"
    assert _reactions(session) == {"r1"}  # r2 gone
    assert "k2" not in _ids(session)  # its orphaned rate constant pruned


def test_set_model_warns_on_dropped_estimated_param(session):
    # k2 is estimated (estimate=1) in the toy parameter table; re-authoring without it drops
    # its row + fitted value, which set_model must warn about.
    res = O.set_model(session, ONE_REACTION_ANTIMONY, justification="drop r2/k2")
    assert res["applied"] is True
    assert any("k2" in w for w in (res.get("warnings") or [])), res.get("warnings")


# -- role enforcement: a shipped-fixed constant must stay fixed -------------
@pytest.fixture
def fixed_k1_session(tmp_path) -> PetabSession:
    """A toy problem that ships k1 (a kinetic constant) FIXED — a stand-in for a conservation
    total / known constant that must not be freed."""
    work = tmp_path / "fixedk1"
    work.mkdir()
    for name in (
        "problem.yaml",
        "conditions.tsv",
        "measurements.tsv",
        "observables.tsv",
        "model.ant",
        "conversion_reaction.xml",
    ):
        src = DATA_DIR / name
        if src.exists():
            shutil.copy(src, work / name)
    (work / "parameters.tsv").write_text(
        "parameterId\tparameterScale\tlowerBound\tupperBound\tnominalValue\testimate\n"
        "k1\tlog10\t0.0001\t100.0\t0.5\t0\n"  # k1 ships FIXED
        "k2\tlog10\t0.0001\t100.0\t0.1\t1\n"
        "sigma_B\tlin\t0.0001\t100.0\t0.1\t0\n"
        "A_init\tlin\t0.0\t100.0\t1.0\t0\n"
        "B_init\tlin\t0.0\t100.0\t0.0\t0\n"
    )
    return PetabSession(
        str(work / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )


def test_shipped_fixed_constant_is_protected(fixed_k1_session):
    sess = fixed_k1_session
    # k1 ships fixed and is a mechanistic (kinetic) constant, so it is protected.
    assert "k1" in sess.protected_fixed_parameter_ids()

    # A (genuine) re-author that KEEPS k1 declared leaves it fixed (reconcile preserves the
    # estimate flag). A structural change — adding a B degradation — so it is not de-duplicated
    # as a no-op re-author of the identical shipped model.
    res = O.set_model(sess, REVERSIBLE_PLUS_DEG_ANTIMONY, justification="keep k1, add degradation")
    assert res["applied"] is True
    k1_row = {r["parameterId"]: r for r in P.parameters_to_records(sess.problem)}["k1"]
    assert k1_row["status"] == "fixed" and k1_row["role"] == "kinetic"


def test_set_model_rejects_removing_a_shipped_fixed_constant(fixed_k1_session):
    from petab_mcp.errors import ValidationError

    sess = fixed_k1_session
    # Re-authoring with only r2 would prune the shipped-fixed constant k1 — a rejected edit.
    with pytest.raises(ValidationError) as exc:
        O.set_model(sess, ONLY_R2_ANTIMONY, justification="drop r1/k1")
    assert any("k1" in issue for issue in (exc.value.issues or [])), exc.value.issues
    # the rejected commit changed nothing: r1 (and k1) are still there
    assert _reactions(sess) == {"r1", "r2"}
    assert "k1" in {r["parameterId"] for r in P.parameters_to_records(sess.problem)}


# -- the model must use every condition-table quantity ----------------------
_DRIVE_MODEL = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_; species B in compartment_; species drive in compartment_;
  A = A_init; B = B_init; drive = drive_init;
  A_init = 1.0; B_init = 0.0; drive_init = 1.0;
  k1 = 0.5; k2 = 0.1;
  r1: A => B; {rate}
  r2: B => A; k2 * B;
end
"""


@pytest.fixture
def drive_condition_session(tmp_path) -> PetabSession:
    """A toy problem whose condition table sets a per-condition input species `drive`."""
    work = tmp_path / "drive"
    work.mkdir()
    for name in (
        "problem.yaml",
        "measurements.tsv",
        "observables.tsv",
        "parameters.tsv",
        "conversion_reaction.xml",
    ):
        shutil.copy(DATA_DIR / name, work / name)
    (work / "conditions.tsv").write_text("conditionId\tdrive\nc0\t1.0\n")
    return PetabSession(
        str(work / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )


def test_set_model_rejects_unused_condition_target(drive_condition_session):
    from petab_mcp.errors import ValidationError

    sess = drive_condition_session
    # `drive` is declared but its value never enters a rate law — the conditions can't perturb
    # the model, so every condition simulates identically. Rejected.
    with pytest.raises(ValidationError) as exc:
        O.set_model(sess, _DRIVE_MODEL.format(rate="k1 * A;"), justification="drive unused")
    assert any("drive" in issue for issue in (exc.value.issues or [])), exc.value.issues


def test_set_model_accepts_used_condition_target(drive_condition_session):
    sess = drive_condition_session
    # `drive` drives r1 — the conditions now actually perturb the model.
    res = O.set_model(
        sess, _DRIVE_MODEL.format(rate="k1 * A * drive;"), justification="drive drives r1"
    )
    assert res["applied"] is True


# -- always-on ICs: estimate a created seed, fix a unit normalization (#90) -------------------
# Authoring on a zero-shot problem (empty parameter table) must add B_init (a positive guess, 460)
# as ESTIMATED — never freezing the guess (the crauste `Naive` failure) — while A == 1.0 is a unit
# normalization and stays FIXED by value.
_ZERO_SHOT_IC_ANTIMONY = """model conversion_reaction
  compartment compartment_ = 1;
  species A in compartment_;
  species B in compartment_;
  A = 1.0;
  B = 460.0;
  k1 = 0.25;
  k2 = 0.05;
  r1: A => B; k1 * A;
  r2: B => A; k2 * B;
end
"""


@pytest.fixture
def zero_shot_empty_ic_session(tmp_path) -> PetabSession:
    """A model-less (zero-shot) problem with an EMPTY parameter table (no shipped ICs), so the ICs
    the agent authors are created fresh by reconcile."""
    import pandas as pd

    work = tmp_path / "empty_ic"
    work.mkdir()
    for name in ("problem.yaml", "conditions.tsv", "observables.tsv", "measurements.tsv"):
        shutil.copy(DATA_DIR / name, work / name)
    _write_empty_sbml(work / "conversion_reaction.xml")
    # Parameter table WITHOUT ICs (reconcile creates them) — matches a zero-shot's empty table;
    # sigma_B stays because the observable's noiseFormula references it (a fixed noise param).
    pdf = pd.read_csv(DATA_DIR / "parameters.tsv", sep="\t")
    pdf[pdf["parameterId"].isin(["k1", "k2", "sigma_B"])].to_csv(
        work / "parameters.tsv", sep="\t", index=False
    )
    return PetabSession(
        str(work / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )


def test_set_model_estimates_created_ic_and_fixes_unit_normalization(zero_shot_empty_ic_session):
    from petab.v1.C import ESTIMATE, LOG10, PARAMETER_SCALE

    sess = zero_shot_empty_ic_session
    res = O.set_model(
        sess, _ZERO_SHOT_IC_ANTIMONY, justification="zero-shot; B seeded at a positive guess"
    )
    assert res["applied"] is True
    pdf = sess.problem.parameter_df
    # B's positive declared IC is only a guess -> ESTIMATED on log10, never frozen (issue #90).
    assert int(pdf.loc["B_init", ESTIMATE]) == 1
    assert pdf.loc["B_init", PARAMETER_SCALE] == LOG10
    # A == 1.0 is a unit normalization -> FIXED by value.
    assert int(pdf.loc["A_init", ESTIMATE]) == 0
    ics = {r["species"]: r for r in res["initial_conditions"]}
    assert ics["B"]["t0_source"] == "estimated"
    assert ics["A"]["t0_source"] == "fixed"
