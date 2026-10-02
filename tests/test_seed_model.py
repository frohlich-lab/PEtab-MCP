"""The content-free initial model a problem that ships WITHOUT one is seeded with.

The empty ``model model / end`` skeleton a model-less problem used to start from is not a
neutral placeholder: it is a broken PEtab problem whose observables reference entities nothing
defines, and AMICI's SBML import SEGFAULTS on it — a process kill no caller can catch. Every
downstream path had to be guarded against reaching it, and v0 could not be restored at all.

Seeding removes the state instead of guarding it: v0 becomes a valid, simulatable, restorable
version that carries no mechanism. These tests pin the two halves that makes true — the seed is
VALID (it lints, it declares what the fixed tables name) and it is CONTENT-FREE (no dynamics, so
``model_authored`` stays false and the shipped parameter table is untouched).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import libsbml
import pytest

from petab_mcp import reconcile as R
from petab_mcp.errors import ValidationError
from petab_mcp.references import StubReferenceResolver
from petab_mcp.session import PetabSession
from petab_mcp.validation import validate_problem

DATA_DIR = Path(__file__).parent / "data" / "toy_v2"


def _model_less_problem(tmp_path: Path) -> Path:
    """A copy of the toy problem whose SBML is EMPTY and which ships no ``model.ant``."""
    work = tmp_path / "empty"
    work.mkdir(parents=True)
    for name in ("problem.yaml", "measurements.tsv", "observables.tsv", "parameters.tsv"):
        shutil.copy(DATA_DIR / name, work / name)
    doc = libsbml.SBMLDocument(3, 2)
    model = doc.createModel()
    model.setId("conversion_reaction")
    model.setMetaId("conversion_reaction")
    libsbml.writeSBMLToFile(doc, str(work / "conversion_reaction.xml"))
    return work / "problem.yaml"


@pytest.fixture
def seeded(tmp_path):
    return PetabSession(
        str(_model_less_problem(tmp_path)),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )


# -- what the seed declares ---------------------------------------------------


def test_the_seed_declares_the_symbols_the_fixed_tables_name(seeded):
    """One species per identifier the observable formulas reference — and nothing else. The toy's
    only observable is `obs_B: B`, and its `sigma_B` already has a parameter row, so `B` is the
    whole seed."""
    assert R.species_ids(seeded.model.sbml_model) == {"B"}


def test_the_seed_has_no_dynamics_so_nothing_reads_as_authored(seeded):
    """Content-free means no reactions and no rules: there IS a model (so AMICI is safe) but no
    mechanism (so the zero-shot contract is unchanged)."""
    sm = seeded.model.sbml_model
    assert sm.getNumReactions() == 0
    assert sm.getNumRules() == 0
    assert seeded.has_model() is True  # something to hand AMICI
    assert seeded.model_authored() is False  # ...but nothing authored
    assert seeded.describe()["model_authored"] is False


def test_the_seed_is_a_valid_petab_problem(seeded):
    """The whole point: v0 LINTS — under the SAME rules every edit is held to, with no exemption.
    A seed that only validated against a relaxed check would just relocate the crash: the next
    parameters-only edit re-validates the working model in full and would fail on it."""
    report = validate_problem(seeded.problem)
    assert report["valid"], report["issues"]


def test_a_function_name_never_becomes_a_species(tmp_path):
    """`log10(B)` names ONE entity, not two. A regex scan of the formula would declare a species
    called `log10`; the seed parses the formula instead (reconcile.formula_identifiers)."""
    yaml_path = _model_less_problem(tmp_path)
    obs = yaml_path.parent / "observables.tsv"
    obs.write_text(obs.read_text().replace("\tB\t", "\tlog10(B)\t"), encoding="utf-8")
    session = PetabSession(
        str(yaml_path), provenance_dir=str(tmp_path / "prov"), resolver=StubReferenceResolver()
    )
    assert R.species_ids(session.model.sbml_model) == {"B"}


# -- what the seed must NOT disturb -------------------------------------------


def test_the_seed_declares_no_parameters_of_its_own(seeded):
    """The seed is species and nothing else. It used to declare every parameter-table row the
    fixed tables did not already resolve, purely so petab would not lint them as "extraneous" —
    which put a pile of parameters the caller never authored into `get_model`, reading as
    structure the seed does not have."""
    sm = seeded.model.sbml_model
    declared = {sm.getParameter(i).getId() for i in range(sm.getNumParameters())}
    # only the parametric IC reconcile minted for the seeded species
    assert declared == {"B_init"}
    assert "k1" not in declared and "k2" not in declared


def test_shipped_fixed_constants_are_still_protected(seeded):
    """`protected_fixed_parameter_ids` reads the SHIPPED table, not v0 — otherwise seeding v0
    (which reconciles it) could quietly empty the set that guards shipped constants."""
    shipped = {str(p) for p in seeded.shipped_parameter_df.index}
    assert {"k1", "k2", "sigma_B", "A_init", "B_init"} <= shipped
    # sigma_B is a noise sigma and *_init are initial conditions, so all three are excluded by
    # design; what matters is that the derivation ran off a table that still had them.
    assert seeded.protected_fixed_parameter_ids() == set()


# -- the seed is working state, not a version ---------------------------------


def test_the_seed_is_working_state_not_a_version(seeded):
    """The seed is valid and simulatable from construction, but it is not a version."""
    assert seeded.provenance.current_version == -1
    assert seeded.provenance.list_versions() == []
    assert seeded.has_model() is True
    assert seeded.model_authored() is False
    assert "B" in seeded.model.to_antimony()


def test_a_model_with_no_dynamics_is_refused(seeded):
    """The other half of the invariant: `make_changes` cannot commit its way BACK to a
    mechanism-less model, so `model_authored` only ever goes false -> true."""
    from petab_mcp import operations as O

    with pytest.raises(ValidationError) as exc:
        O.set_model(
            seeded,
            "model m\n  species B;\n  B = 0;\nend",
            justification="declare B but no reaction",
        )
    assert "no DYNAMICS" in str(exc.value)
    assert seeded.model_authored() is False  # and the working model is untouched


def test_a_petab_placeholder_never_becomes_a_species(tmp_path):
    """`observableParameter1_<obsId>` is substituted from the measurement table at import; it is
    not an entity, and whatever it resolves to already has a parameter row. Declaring it would
    invent a state the real model never has."""
    yaml_path = _model_less_problem(tmp_path)
    obs = yaml_path.parent / "observables.tsv"
    obs.write_text(
        "observableId\tobservableFormula\tobservablePlaceholders\tnoiseFormula\n"
        "obs_B\tobservableParameter1_obs_B * B\tobservableParameter1_obs_B\tsigma_B\n",
        encoding="utf-8",
    )
    meas = yaml_path.parent / "measurements.tsv"
    lines = meas.read_text().splitlines()
    header = lines[0] + "\tobservableParameters"
    rows = [f"{ln}\t1.0" for ln in lines[1:] if ln.strip()]
    meas.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")

    session = PetabSession(
        str(yaml_path), provenance_dir=str(tmp_path / "prov"), resolver=StubReferenceResolver()
    )
    assert R.species_ids(session.model.sbml_model) == {"B"}


def test_the_seed_survives_a_parameters_only_edit(tmp_path, add_condition):
    """The seed is not just valid AT GENERATION — it stays valid under the full check every edit
    re-runs. A bounds edit made while the seed is still the working model re-validates it, so a
    seed that only passed a relaxed check would fail the caller's very first parameter change
    (and again after `restore_version(0)`). Uses a problem WITH a condition target, which is the
    case that breaks: the seed declares the target so petab lints, but has no mechanism to
    consume it."""
    from petab_mcp import operations as O

    yaml_path = _model_less_problem(tmp_path)
    add_condition(yaml_path, "S", 2.0)

    session = PetabSession(
        str(yaml_path), provenance_dir=str(tmp_path / "prov"), resolver=StubReferenceResolver()
    )
    assert "S" in R.species_ids(session.model.sbml_model)
    assert validate_problem(session.problem)["valid"]

    res = O.change_parameters(
        session,
        [{"parameter_id": "sigma_B", "lower_bound": 1e-3, "upper_bound": 1.0}],
        justification="tighten the noise box before authoring",
    )
    assert res["applied"] is True


def test_an_authored_model_must_still_consume_its_condition_targets(tmp_path, add_condition):
    """The skip is scoped to "no dynamics", not to "the seed": a real mechanism that ignores a
    condition target is still rejected, because `make_changes` refuses a dynamics-free model in
    the first place — so nothing a caller authors can slip through the gap."""
    from petab_mcp import operations as O

    yaml_path = _model_less_problem(tmp_path)
    add_condition(yaml_path, "S", 2.0)
    session = PetabSession(
        str(yaml_path), provenance_dir=str(tmp_path / "prov"), resolver=StubReferenceResolver()
    )
    assert {"B", "S"} <= R.species_ids(session.model.sbml_model)

    with pytest.raises(ValidationError) as exc:
        O.set_model(
            session,
            "model m\n  species B; species S;\n  B = 0; S = 0;\n  kb = 0.1;\n  r1: -> B; kb;\nend",
            justification="mechanism ignoring S",
        )
    assert any("never uses 'S'" in i for i in exc.value.issues), exc.value.issues


# -- which ids can actually be declared ---------------------------------------


def test_a_legal_id_python_reserves_is_still_seeded(tmp_path):
    """`lambda` is a perfectly good SBML/Antimony id. The filter used to be
    `keyword.iskeyword` — the wrong language's reserved list — so an observable naming it made
    the seed omit the species, fail to lint, and silently fall back to the broken skeleton."""
    yaml_path = _model_less_problem(tmp_path)
    obs = yaml_path.parent / "observables.tsv"
    obs.write_text(obs.read_text().replace("\tB\t", "\tlambda\t"), encoding="utf-8")
    params = yaml_path.parent / "parameters.tsv"
    params.write_text(params.read_text().replace("\nB_init\t", "\nlambda_init\t"), encoding="utf-8")

    session = PetabSession(
        str(yaml_path), provenance_dir=str(tmp_path / "prov"), resolver=StubReferenceResolver()
    )
    assert "lambda" in R.species_ids(session.model.sbml_model)
    assert validate_problem(session.problem)["valid"]


def test_the_reserved_word_list_matches_libantimony():
    """`_ANTIMONY_RESERVED` was derived by probing libantimony, not read off a spec, so it is
    pinned against libantimony here: a release that adds or drops a grammar keyword shows up as a
    failure rather than as a seed that silently drops a species (too strict) or emits Antimony
    that will not parse (too loose).

    The MathML constants (`time`, `pi`, ...) are deliberately absent from that set —
    `_declarable_id` asks libSBML about those — so the check is on the COMBINED verdict."""
    import antimony

    def _antimony_accepts(sid: str) -> bool:
        antimony.clearPreviousLoads()
        if antimony.loadAntimonyString(f"model probe\n  species {sid};\n  {sid} = 0;\nend") < 0:
            return False
        sbml = antimony.getSBMLString(antimony.getMainModuleName())
        model = libsbml.readSBMLFromString(sbml).getModel()
        return [model.getSpecies(i).getId() for i in range(model.getNumSpecies())] == [sid]

    probes = sorted(
        R._ANTIMONY_RESERVED
        # MathML reserved constants: libSBML's job, not the list's.
        | {"time", "pi", "avogadro", "exponentiale", "true", "false", "inf", "nan"}
        # ...and ids that must stay usable, including the one the old filter wrongly rejected.
        | {"B", "S_dose", "lambda", "encodes", "identity", "part", "default", "x2"}
    )
    disagree = [sid for sid in probes if R._declarable_id(sid) != _antimony_accepts(sid)]
    assert not disagree, f"filter disagrees with libantimony on: {disagree}"


# -- lint noise ----------------------------------------------------------------


def test_design_only_lint_warnings_never_reach_the_caller(seeded):
    """petab warns about a condition or experiment no measurement uses — a statement about the
    FIXED design (fujita ships its held-out pulses in the condition table), which no call can
    change. Promoting it would put a fixed, unactionable string on the warnings of EVERY
    make_changes result."""
    import pandas as pd

    seeded.problem.condition_df = pd.DataFrame(
        {"conditionId": ["c_unused"], "targetId": ["B"], "targetValue": [1.0]}
    )
    seeded.problem.experiment_df = pd.DataFrame(
        {"experimentId": ["e_unused"], "time": [0.0], "conditionId": ["c_unused"]}
    )
    lint = {i.task for i in seeded.problem.to_petab().validate()}
    assert "CheckUnusedExperiments" in lint  # petab does warn...
    report = validate_problem(seeded.problem)
    assert report["n_warnings"] == 0, report["issues"]  # ...and the caller never sees it


# -- the seed is reconciled like any other model -------------------------------


def test_the_seed_is_reconciled_and_its_initial_conditions_are_estimated(tmp_path):
    """The seed goes through `reconcile` like anything else, so its species get parametric ICs.
    That only works because they are seeded at a POSITIVE value: at 0 (or 1) the leak-free
    convention would hold the minted `<species>_init` FIXED, and reconcile never re-touches an
    existing row — so the caller's first authoring would inherit a frozen initial condition,
    which is the trap of #90 on the very species it is about to build a mechanism around."""
    from petab.v2.C import ESTIMATE
    from petab_mcp.problem import LOG10, PARAMETER_SCALE

    yaml_path = _model_less_problem(tmp_path)
    # Ship no IC rows, so the seed's reconcile is what mints them.
    params = yaml_path.parent / "parameters.tsv"
    kept = [ln for ln in params.read_text().splitlines() if not ln.startswith(("A_init", "B_init"))]
    params.write_text("\n".join(kept) + "\n", encoding="utf-8")

    session = PetabSession(
        str(yaml_path), provenance_dir=str(tmp_path / "prov"), resolver=StubReferenceResolver()
    )
    pdf = session.problem.parameter_df
    assert "B_init" in pdf.index  # reconcile minted it
    assert int(pdf.loc["B_init", ESTIMATE]) == 1  # ...ESTIMATED, not frozen
    assert pdf.loc["B_init", PARAMETER_SCALE] == LOG10
    assert int(session.problem.n_estimated) > 0  # so the seed is fittable


def test_a_shipped_row_comes_back_as_the_problem_defined_it(tmp_path):
    """The seed declares no mechanism, so every kinetic / initial-condition row a zero-shot
    problem ships is pruned at v0. That is only safe because re-declaring the parameter RESTORES
    the shipped row — bounds, scale and estimate — rather than minting defaults, so the candidate
    and the reference are fit over the same box. A shipped PRIOR is not part of the row: the
    objective is the likelihood the criterion scores, so the table does not carry one at all."""
    from petab_mcp import operations as O

    yaml_path = _model_less_problem(tmp_path)
    params = yaml_path.parent / "parameters.tsv"
    # k1 ships a distinctive box + a prior; the box must come back, the prior must not.
    rows = params.read_text().rstrip().splitlines()
    header = rows[0] + "\tpriorDistribution\tpriorParameters"
    body = [(r + "\tnormal\t0.5;0.2") if r.startswith("k1\t") else (r + "\t\t") for r in rows[1:]]
    params.write_text("\n".join([header, *body]) + "\n", encoding="utf-8")

    session = PetabSession(
        str(yaml_path), provenance_dir=str(tmp_path / "prov"), resolver=StubReferenceResolver()
    )
    assert "k1" not in session.problem.parameter_df.index  # pruned: the seed has no mechanism

    O.set_model(
        session,
        "model m\n  species B;\n  B = 1;\n  k1 = 0.5;\n  r1: B => ; k1 * B;\nend",
        justification="author a mechanism that uses the shipped k1",
    )
    row = session.problem.parameter_df.loc["k1"]
    assert float(row["lowerBound"]) == 1e-4  # the SHIPPED box, not _default_bounds
    assert float(row["upperBound"]) == 100.0
    assert "priorDistribution" not in session.problem.parameter_df.columns  # never carried


def test_a_pruned_shipped_parameter_is_still_shown_to_the_caller(seeded):
    """Pruning a shipped row must not HIDE it. `k1`/`k2` are parameters the problem HANDED the
    caller — at v0 they have no row (the seed authors no mechanism to use them), but a caller
    that cannot see them cannot know what it is meant to build on. They come back from
    `get_parameters` flagged `in_model: false`, carrying the values the problem gave them, and
    the rows the model DOES declare are flagged `in_model: true`."""
    from petab_mcp import parameters as P

    records = P.parameters_to_records(seeded.problem, shipped=seeded.shipped_parameter_df)
    by_id = {r["parameterId"]: r for r in records}

    assert {"k1", "k2"} <= set(by_id)  # visible...
    assert "k1" not in seeded.problem.parameter_df.index  # ...but not in the live table
    assert by_id["k1"]["in_model"] is False
    assert by_id["k1"]["role"] == "constant"  # classified even without a row
    assert by_id["k1"]["status"] == "estimated"  # ...and it restates the shipped estimate flag
    assert float(by_id["k1"]["lowerBound"]) == 1e-4  # the values the PROBLEM defined
    assert float(by_id["k1"]["upperBound"]) == 100.0

    assert by_id["sigma_B"]["in_model"] is True  # used by the fixed noise formula: kept
    assert by_id["B_init"]["in_model"] is True  # the seed's own parametric IC


def test_a_re_declared_parameter_stops_being_flagged_absent(tmp_path):
    """`in_model` tracks the live table, so authoring a mechanism that uses `k1` flips it — the
    caller sees the same id move from "shipped, unused" to "in your model"."""
    from petab_mcp import operations as O
    from petab_mcp import parameters as P

    session = PetabSession(
        str(_model_less_problem(tmp_path)),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )
    O.set_model(
        session,
        "model m\n  species B;\n  B = 1;\n  k1 = 0.5;\n  r1: B => ; k1 * B;\nend",
        justification="use the shipped k1",
    )

    by_id = {
        r["parameterId"]: r
        for r in P.parameters_to_records(session.problem, shipped=session.shipped_parameter_df)
    }
    assert by_id["k1"]["in_model"] is True
    assert by_id["k1"]["role"] == "kinetic"  # now it IS a rate constant
    assert by_id["k2"]["in_model"] is False  # still unused
    assert sum(1 for r in by_id.values() if r["parameterId"] == "k1") == 1  # not duplicated


def test_a_problem_with_no_declarable_entity_refuses_to_construct(tmp_path):
    """The invariant is ENFORCED, not assumed. A seed can lint perfectly and still declare no
    SPECIES — an observable formula that is a bare constant names no entity — which lands right
    back in the empty state the guards used to cover. Nothing downstream would notice until
    AMICI segfaulted, so construction refuses."""
    yaml_path = _model_less_problem(tmp_path)
    (yaml_path.parent / "observables.tsv").write_text(
        "observableId\tobservableFormula\tnoiseFormula\nobs_B\t1.0\tsigma_B\n", encoding="utf-8"
    )

    with pytest.raises(ValidationError, match="nothing that could be declared"):
        PetabSession(
            str(yaml_path), provenance_dir=str(tmp_path / "prov"), resolver=StubReferenceResolver()
        )
