"""A readout constant in time where its data change, named on the fit, and the enzyme taken in
every state that makes one, named on the edit (PEtab-MCP#232)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest
from indra.statements import Activation, Agent, Dephosphorylation, IncreaseAmount, Phosphorylation

from petab_mcp import data_access as D
from petab_mcp import problem_sbml as PS
from petab_mcp.mechanisms import INDRA_DEFAULT
from petab_mcp.references import StubReferenceResolver
from petab_mcp.server import create_server


def _call(mcp, name, args):
    return asyncio.run(mcp.call_tool(name, args)).structured_content


def _server(tmp_path, empty_yaml):
    """The toy's `obs_B` (`B`, rising 0 -> 0.79) as `S{p}`, measured once `dose` goes from 0 in
    pre-equilibration to 1, with no model until the rows build one."""
    work = Path(empty_yaml).parent
    (work / "conditions.tsv").write_text(
        "conditionId\ttargetId\ttargetValue\npre\tdose\t0\ntreat\tdose\t1\n", encoding="utf-8"
    )
    (work / "experiments.tsv").write_text(
        "experimentId\ttime\tconditionId\ne0\t-inf\tpre\ne0\t0\ttreat\n", encoding="utf-8"
    )
    meas = work / "measurements.tsv"
    meas.write_text(meas.read_text().replace("obs_B\t\t", "obs_B\te0\t"), encoding="utf-8")
    yaml = work / "problem.yaml"
    yaml.write_text(
        yaml.read_text()
        + "condition_files:\n- conditions.tsv\nexperiment_files:\n- experiments.tsv\n"
    )
    (tmp_path / "entities.json").write_text(
        '{"B": {"name": "S", "mods": [{"mod_type": "phosphorylation"}]}}'
    )
    return create_server(
        str(yaml),
        provenance_dir=str(tmp_path / "prov"),
        resolver=StubReferenceResolver(),
        entity_map=str(tmp_path / "entities.json"),
    )


def _build(mcp, enzyme):
    return _call(
        mcp,
        "make_changes",
        {
            "justification": "the dose activates E, which phosphorylates S",
            "statements": [
                "r1: K phosphorylates E",
                f"r2: {enzyme} phosphorylates S",
                "r3: S is dephosphorylated",
            ],
            "bindings": {"r1": "default", "r2": "default", "r3": "basal"},
            "seeds": [{"agent": "K", "amount": "dose"}],
            "n_starts": 2,
        },
    )


def test_a_bare_enzyme_is_named_on_the_edit_and_its_flat_readout_on_the_fit(tmp_path, empty_yaml):
    """raimundez r2's `MEK phosphorylates MAPK` read MEK{u} + MEK{p}, a conserved total: pMAPK was
    flat in 13/13 experiments across 18 versions, and no reply said why. Through `make_changes`,
    the fit and `get_fit_diagnosis`, as the agent meets it."""
    mcp = _server(tmp_path, empty_yaml)
    out = _build(mcp, "E")
    assert out["applied"] is True, out.get("error")
    said = " ".join(m["message"] for m in out["messages"])
    assert "Row `r2` names no state of E, its catalyst" in said
    assert "name its state: `E{p} phosphorylates S`" in said
    assert "nothing that changes E's state moves the rate" in said
    structure = out["residual_structure"]
    assert structure["constant_in_time"] == [{"observable_id": "obs_B", "experiments": "1/1"}]
    assert "Constant in time while the data change: obs_B (1/1 experiments)" in structure["summary"]
    diagnosis = _call(mcp, "get_fit_diagnosis", {})
    assert diagnosis["constant_in_time"] == structure["constant_in_time"]

    fixed = _build(mcp, "E{p}")
    assert fixed["applied"] is True, fixed.get("error")
    assert not any("catalyst" in m["message"] for m in fixed["messages"])
    assert "constant_in_time" not in fixed["residual_structure"]


def _notes(rows):
    notes: list[str] = []
    PS.to_antimony(rows, {"B": Agent("S")}, notes=notes)
    return [n for n in notes if "catalyst" in n]


_SUB = (Dephosphorylation(None, Agent("S")), "basal", "r3")


@pytest.mark.parametrize(
    ("rows", "form", "constant"),
    [
        # Its activity, not a modification, is what another row changes.
        ([(Activation(Agent("K"), Agent("E")), INDRA_DEFAULT, "r1")], "E{activity:on}", True),
        # A row makes E: its total moves, so the rate is not constant, only blind to E's state.
        (
            [
                (Phosphorylation(Agent("K"), Agent("E")), INDRA_DEFAULT, "r1"),
                (IncreaseAmount(None, Agent("E")), INDRA_DEFAULT, "r4"),
            ],
            "E{p}",
            False,
        ),
    ],
)
def test_the_state_another_row_gives_the_enzyme_is_the_one_offered(rows, form, constant):
    """The form offered is the state another row puts on the enzyme, and a constant rate is
    claimed only where nothing makes or removes it; a synthesis, which puts E's first state on a
    new molecule, changes no state."""
    (note,) = _notes([*rows, (Phosphorylation(Agent("E"), Agent("S")), INDRA_DEFAULT, "r2"), _SUB])
    assert f"`{form} phosphorylates S`" in note and "changes in `r1`." in note
    assert ("its total is constant" in note) is constant


def test_an_enzyme_no_row_changes_says_nothing():
    """A state site only a seed gives (`E{p}` declared, never put by a row) is not taken in every
    state: no row makes it a different molecule over time."""
    p_e = Agent._from_json({"name": "E", "mods": [{"mod_type": "phosphorylation"}]})
    notes: list[str] = []
    PS.to_antimony(
        [(Phosphorylation(Agent("E"), Agent("S")), INDRA_DEFAULT, "r2"), _SUB],
        {"B": Agent("S")},
        seeds=[("E_p", p_e, 1.0), ("S", Agent("S"), 1.0)],
        notes=notes,
    )
    assert not [n for n in notes if "catalyst" in n]


def _series(simulation, measurement, noise=0.1):
    return pd.DataFrame(
        {
            "observable_id": "y",
            "experiment_id": "e0",
            "time": range(len(measurement)),
            "simulation": simulation,
            "measurement": measurement,
            "residual_weighted": [(m - s) / noise for s, m in zip(simulation, measurement)],
        }
    )


@pytest.mark.parametrize(
    ("measurement", "named"),
    [
        ([0.0, 0.3, 0.48, 0.66, 0.72, 0.79], True),
        # Noise about one level, as a total protein's data are: a flat line is right there.
        ([0.5, 0.55, 0.42, 0.51, 0.47, 0.56], False),
    ],
)
def test_only_data_that_change_beyond_their_noise_name_a_flat_readout(measurement, named):
    """A flat simulation is named where its data change, and not where they scatter about one
    level within their noise: that would repeat on every fit of a correct model."""
    flat = D.constant_in_time(_series([0.5] * 6, measurement))
    assert flat == ([{"observable_id": "y", "experiments": "1/1"}] if named else [])
