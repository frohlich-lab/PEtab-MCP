"""The `statements` extra: INDRA, pysb and BioNetGen are the statements route's alone, so a server
without an entity map runs without them, and one with an entity map refuses to start without them,
naming what to install."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest

from petab_mcp import cli
from petab_mcp.errors import ValidationError
from petab_mcp.server import STATEMENTS_EXTRA, create_server


def _without(monkeypatch, *names: str) -> None:
    """Make ``importlib.util.find_spec`` report ``names`` as not installed."""
    find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util, "find_spec", lambda name, *a: None if name in names else find_spec(name, *a)
    )


def test_a_server_without_an_entity_map_starts_without_the_extra(tmp_path, yaml_path):
    """In a process where the extra's packages cannot be imported, as where they are not installed
    (AMICI's PEtab importer tries pysb and goes on without it)."""
    code = textwrap.dedent(
        f"""
        import sys
        from petab_mcp.server import STATEMENTS_EXTRA

        class NotInstalled:
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in STATEMENTS_EXTRA:
                    raise ModuleNotFoundError(f"No module named {{name!r}}", name=name)

        sys.meta_path.insert(0, NotInstalled())
        from petab_mcp.references import StubReferenceResolver
        from petab_mcp.server import create_server

        create_server(
            {yaml_path!r},
            provenance_dir={str(tmp_path / "prov")!r},
            build_root={str(tmp_path / "build")!r},
            resolver=StubReferenceResolver(),
            auto_fit=False,
        )
        print("started")
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0 and out.stdout.strip().endswith("started"), out.stderr[-2000:]


def test_an_entity_map_without_the_extra_is_refused_naming_what_to_install(monkeypatch, yaml_path):
    _without(monkeypatch, "pysb")
    with pytest.raises(ValidationError, match=r"pysb is not installed.*petab-mcp\[statements\]"):
        create_server(yaml_path, entity_map="map.json", auto_fit=False)


def test_the_cli_refuses_it_while_its_stderr_is_still_the_terminals(monkeypatch, capsys):
    _without(monkeypatch, *STATEMENTS_EXTRA)
    monkeypatch.setattr(cli, "_send_stderr_to", lambda path: pytest.fail("stderr redirected"))
    assert cli.main(["--petab-yaml", "problem.yaml", "--entity-map", "map.json"]) == 2
    err = capsys.readouterr().err
    assert "indra, pysb, bionetgen are not installed" in err and "petab-mcp[statements]" in err
