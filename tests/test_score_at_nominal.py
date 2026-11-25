"""Scoring a problem that arrives with its model already finished.

Every other path through this server judges a model the AGENT built through `make_changes`, which
always fits before it snapshots. A problem can also arrive complete — delivered by something that
did its own fitting elsewhere — and then the parameter table's `nominalValue`s ARE the answer
being judged. Refitting would score a different model than the one delivered, so this path
simulates only and reports those nominal values as the fitted point.

Reporting them matters as much as the score: `noise_sigma_report` and the at-bound check read
`fitted_parameters`, and an empty one would quietly exempt the delivery from those gates.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

pytest.importorskip("amici", reason="compiles a model")

TOY = Path(__file__).parent / "data" / "toy_v2"


def test_score_shipped_model_reports_the_nominal_point(amici_build_root, tmp_path):
    """`create_server(score_shipped_model=True)` compiles the shipped toy and scores it at its
    table's nominal values, reporting them as the fitted point so the sigma gate and the
    at-bound check have something to read. (Compiles AMICI — run under srun.)"""
    from petab_mcp.server import create_server
    from petab_mcp.references import StubReferenceResolver

    work = tmp_path / "work"
    shutil.copytree(TOY, work)
    create_server(
        str(work / "problem.yaml"),
        build_root=str(amici_build_root),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        score_shipped_model=True,
    )
    # create_server returns only the FastMCP app; build a session+engine and score at nominal directly.
    from petab_mcp.session import PetabSession
    from petab_mcp.engine import AmiciEngine
    from petab_mcp import fitting

    session = PetabSession(
        str(work / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov2"),
        resolver=StubReferenceResolver(),
    )
    engine = AmiciEngine(session, build_root=str(amici_build_root))
    session.engine = engine
    payload = fitting.score_at_nominal(session, engine)
    assert payload["status"] == "scored_at_nominal"
    assert set(payload["fitted_parameters"]) == {"k1", "k2"}
    assert payload["fitted_parameters"]["k1"] == pytest.approx(0.5)
    assert payload["parameter_scales"] == {"k1": "log10", "k2": "log10"}
    assert payload["neg_log_likelihood"] is not None and payload["n_data_points"] > 0
    summary = fitting.fit_summary(session, engine)
    assert summary["neg_log_likelihood"] == payload["neg_log_likelihood"]
    assert summary["bic"] is not None  # k came from the table, n from the simulation


def test_a_delivered_model_of_rate_rules_alone_is_scored(amici_build_root, tmp_path):
    """A delivery whose states are all rate-rule PARAMETERS (no species, no reactions) is the
    toy's own mechanism written another way, so it scores what the toy scores at the same nominal
    values. The session used to seed over it and report it as never built. (Compiles AMICI —
    run under srun.)"""
    from petab_mcp import fitting
    from petab_mcp.engine import AmiciEngine
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession
    from test_seed_model import rules_only_problem

    def score(yaml_path, prov):
        session = PetabSession(
            str(yaml_path), provenance_dir=str(prov), resolver=StubReferenceResolver()
        )
        engine = AmiciEngine(session, build_root=str(amici_build_root))
        session.engine = engine
        return fitting.score_at_nominal(session, engine)

    work = tmp_path / "toy"
    shutil.copytree(TOY, work)
    reactions = score(work / "problem.yaml", tmp_path / "prov_toy")
    rules = score(rules_only_problem(tmp_path), tmp_path / "prov_rules")
    assert rules["status"] == "scored_at_nominal"
    assert rules["neg_log_likelihood"] == pytest.approx(reactions["neg_log_likelihood"], rel=1e-6)


def _delivered_toy(tmp_path, amici_build_root):
    """The toy with `sigma_B` (0.3) and a scaling `s_B` (2.0) both estimated through placeholders,
    so the fit's hierarchical objective inner-solves both; ``(session, engine)``."""
    import pandas as pd

    from petab_mcp.engine import AmiciEngine
    from petab_mcp.references import StubReferenceResolver
    from petab_mcp.session import PetabSession

    work = tmp_path / "work"
    shutil.copytree(TOY, work)
    (work / "observables.tsv").write_text(
        "observableId\tobservableFormula\tnoiseFormula\tobservablePlaceholders\tnoisePlaceholders\n"
        "obs_B\tobservableParameter1_obs_B * B\tnoiseParameter1_obs_B\t"
        "observableParameter1_obs_B\tnoiseParameter1_obs_B\n"
    )
    meas = pd.read_csv(work / "measurements.tsv", sep="\t", keep_default_na=False)
    meas.assign(observableParameters="s_B", noiseParameters="sigma_B").to_csv(
        work / "measurements.tsv", sep="\t", index=False
    )
    pars = work / "parameters.tsv"
    pars.write_text(
        pars.read_text().replace("0.1\tfalse", "0.3\ttrue")  # sigma_B's row
        + "s_B\tlin\t0.01\t100.0\t2.0\ttrue\n"
    )
    session = PetabSession(
        str(work / "problem.yaml"),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
    )
    engine = AmiciEngine(session, build_root=str(amici_build_root))
    session.engine = engine
    return session, engine


def test_a_delivered_sigma_and_scaling_are_scored_at_their_delivered_values(
    amici_build_root, tmp_path
):
    """The fit's hierarchical objective inner-solves an estimated sigma and scaling. Scoring a
    delivery through it re-solved both instead of reading the delivered values; for the sigma
    alone that reported 10.7 deviance units better than the delivered model (#8)."""
    import math

    from petab_mcp import fitting

    session, engine = _delivered_toy(tmp_path, amici_build_root)
    payload = fitting.score_at_nominal(session, engine)
    assert payload["fitted_parameters"]["sigma_B"] == pytest.approx(0.3)
    assert payload["fitted_parameters"]["s_B"] == pytest.approx(2.0)
    at_1 = next(r for r in payload["records"] if r["time"] == 1.0)
    assert at_1["simulation"] == pytest.approx(2.0 * 5 / 6 * (1 - math.exp(-0.6)), rel=1e-4)
    weighted = [r["residual_weighted"] for r in payload["records"]]
    at_table = sum(0.5 * w * w + 0.5 * math.log(2 * math.pi * 0.3**2) for w in weighted)
    assert payload["neg_log_likelihood"] == pytest.approx(at_table, rel=1e-6)

    hierarchical, inner = fitting.estimation_problem(session, engine)
    assert sorted(inner) == ["s_B", "sigma_B"], "the fit would inner-solve both"
    x = fitting.problem_x_from_nominal(hierarchical, session.problem.parameter_df)
    assert float(hierarchical.objective(x)) < payload["neg_log_likelihood"] - 1


def test_a_lost_inner_write_back_fails_the_fit_and_a_stale_table_is_noted(
    amici_build_root, tmp_path, monkeypatch
):
    """A pass with no fit scores the table, so the table must hold the fit's inner optimum. A fit
    whose inner values do not come back from pyPESTO fails rather than keep the pre-fit sigma and
    scaling; and a version whose table still does not reproduce its committed score says so, on
    restore and on read."""
    from petab_mcp import fitting
    from petab_mcp.errors import ModelUpdateError

    session, engine = _delivered_toy(tmp_path, amici_build_root)
    table = session.problem.parameter_df.copy()
    # The fit reuses this cached problem; pyPESTO reads the ids while building it, so it is patched
    # once built.
    hierarchical, _ = fitting.estimation_problem(session, engine)
    monkeypatch.setattr(
        hierarchical.objective.calculator, "get_interpretable_inner_par_ids", lambda: []
    )
    with pytest.raises(ModelUpdateError, match="s_B, sigma_B"):
        session.fit(n_starts=1, justification="test")
    monkeypatch.undo()
    assert session.problem.parameter_df.equals(table), "nothing was written back"

    # A version stored without its simulation, committed with a score its table does not give.
    version = session.provenance.snapshot_version(
        model_ant=session.model.to_antimony(),
        parameters=session.problem.parameter_df,
        kind="fit",
        meta={"neg_log_likelihood": -50.0},
    )
    session.rollback_to(version)
    assert "committed with nll -50" in session.simulation["objective_note"]

    session.simulation = None
    fitting.ensure_simulation(session, engine)
    assert "committed with nll -50" in session.simulation["objective_note"]
