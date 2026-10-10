"""Scoring a problem that arrives with its model already finished.

Every other path through this server judges a model the AGENT built through `make_changes`, which
always fits before it snapshots. A problem can also arrive complete — delivered by something that
did its own fitting elsewhere — and then the parameter table's `nominalValue`s ARE the answer
being judged. Refitting would score a different model than the one delivered, so this path
simulates only and reports those nominal values as the fitted point.

Reporting them matters as much as the score: `noise_sigma_report` and the at-bound check read
`fitted_parameters`, and an empty one would quietly exempt the delivery from the judge's gates.
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
