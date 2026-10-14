"""Where the CLI server's stdout and stderr go (``cli._send_stderr_to``,
``cli._reserve_stdout_for_the_protocol``).

Nobody reads a server's stderr, so it goes to ``<store>/stderr.log``. Under stdio, fd 1 is the
JSON-RPC channel. AMICI ``print()``s a failed build's whole compiler log to stdout (55 lines for a
missing compiler alone, more for a real compile error), and native code, compiler subprocesses and
pyPESTO workers inherit fd 1. All of that must reach the file instead.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from fastmcp import Client
from fastmcp.client.transports import StdioTransport


def test_only_the_protocol_reaches_stdout_and_only_a_pointer_reaches_stderr(tmp_path):
    """A ``print``, raw writes to fd 1 and fd 2, and a child process (like a compiler or a pyPESTO
    worker) all land in the file. The real stdout gets only what went through
    ``sys.stdout.buffer``, which is what the MCP SDK writes to. The real stderr gets only the line
    saying where everything else went."""
    log = tmp_path / "store" / "stderr.log"
    child = "import sys; print('child out'); print('child err', file=sys.stderr)"
    body = textwrap.dedent(f"""
        import os, subprocess, sys
        from pathlib import Path
        from petab_mcp import cli
        cli._send_stderr_to(Path({str(log)!r}))
        cli._reserve_stdout_for_the_protocol()
        print("from print")
        os.write(1, b"from fd 1\\n")
        os.write(2, b"from fd 2\\n")
        subprocess.run([sys.executable, "-c", {child!r}], check=True)
        sys.stdout.buffer.write(b"protocol\\n")
        sys.stdout.buffer.flush()
    """)
    out = subprocess.run([sys.executable, "-c", body], capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, log.read_text() if log.exists() else out.stderr
    assert out.stdout == "protocol\n"
    assert f"stderr goes to {log}" in out.stderr
    text = log.read_text()
    for line in ("from print", "from fd 1", "from fd 2", "child out", "child err"):
        assert line in text
        assert line not in out.stderr


def test_a_stdio_server_answers_through_the_reserved_channel(tmp_path):
    """The stdout guard rests on one SDK detail: the stdio transport wraps ``sys.stdout.buffer``
    once, when it starts. If it wrote ``sys.stdout`` instead, every reply would go to the log and
    the server would never answer, so this drives a real server over a real pipe."""
    shutil.copytree(Path(__file__).parent / "data" / "toy_v2", tmp_path / "p")
    client_view = tmp_path / "server_stderr.txt"
    transport = StdioTransport(
        sys.executable,
        ["-m", "petab_mcp.cli", "--petab-yaml", str(tmp_path / "p" / "problem.yaml")],
        env=dict(os.environ),  # the SDK otherwise passes a minimal environment
        keep_alive=False,
        log_file=client_view,
    )

    async def go():
        async with Client(transport, timeout=60, init_timeout=60) as client:
            return await client.call_tool("describe_problem", {})

    assert asyncio.run(go()).structured_content
    assert (tmp_path / "p" / ".petab_mcp" / "stderr.log").is_file()
    assert "stderr goes to" in client_view.read_text()
