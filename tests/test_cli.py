"""Where the CLI server's stdout and stderr go (``cli._send_stderr_to``,
``cli._reserve_stdout_for_the_protocol``), and the SLURM pool it fits on (``cli._slurm_pool``).

Nobody reads a server's stderr, so it goes to ``<store>/stderr.log``. Under stdio, fd 1 is the
JSON-RPC channel. AMICI ``print()``s a failed build's whole compiler log to stdout (55 lines for a
missing compiler alone, more for a real compile error), and native code, compiler subprocesses and
pyPESTO workers inherit fd 1. All of that must reach the file instead.

CI has no SLURM, so the pool tests never submit a job.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import dask
import dask_jobqueue
import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from petab_mcp import cli


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


def test_the_pool_takes_every_slurm_setting_from_dask_config(tmp_path):
    """``_slurm_pool`` gives ``SLURMCluster`` no settings, so what the README's ``jobqueue.yaml``
    sets must reach the job script from dask's config alone: partition, cores, one single-threaded
    worker process per core, memory, walltime, the workers' lifetime and both directories. The
    scheduler starts, but with a minimum of 0 jobs and no work nothing is submitted."""
    logs = tmp_path / "logs"
    config = {
        "jobqueue.slurm.queue": "ncpu",
        "jobqueue.slurm.cores": 4,
        "jobqueue.slurm.processes": 4,
        "jobqueue.slurm.memory": "8GiB",
        "jobqueue.slurm.walltime": "01:00:00",
        "jobqueue.slurm.worker-extra-args": ["--lifetime", "55m"],
        "jobqueue.slurm.local-directory": "$TMPDIR",
        "jobqueue.slurm.log-directory": str(logs),
        "jobqueue.slurm.scheduler-options": {"host": "127.0.0.1"},  # no worker will connect
    }
    # Every job reads the config as it is described, the script's one included.
    with dask.config.set(config), cli._slurm_pool(min_jobs=0, max_jobs=2) as cluster:
        script = cluster.job_script()
    for directive in ("-p ncpu", "--cpus-per-task=4", "--mem=8G", "-t 01:00:00", f"-o {logs}/"):
        assert f"#SBATCH {directive}" in script
    for arg in ("--nworkers 4", "--nthreads 1", "--lifetime 55m", "--local-directory $TMPDIR"):
        assert arg in script


def _never(*args):
    raise AssertionError("main went past its refusal")


_BOUNDS = "0 <= --slurm-min-jobs <= --slurm-max-jobs"


@pytest.mark.parametrize(
    "flags, says",
    [
        (["--slurm-max-jobs", "2"], "--build-root"),
        (["--slurm-max-jobs", "-1"], _BOUNDS),
        (["--slurm-min-jobs", "1"], _BOUNDS),
        (["--slurm-max-jobs", "2", "--slurm-min-jobs", "3", "--build-root", "b"], _BOUNDS),
        (["--slurm-max-jobs", "2", "--slurm-min-jobs", "-1", "--build-root", "b"], _BOUNDS),
    ],
)
def test_a_pool_main_cannot_run_is_refused_before_anything_starts(monkeypatch, capsys, flags, says):
    """The workers import the compiled models by the path the server built them at, and the default
    build root is a temp dir on the server's own node, which no worker can read. Job bounds outside
    0 <= min <= max would otherwise pass without a word: a negative maximum fits on this node, a
    minimum without a maximum starts no pool, and dask clamps the rest. ``main`` refuses each at
    once, while its stderr is still the terminal's."""
    monkeypatch.delenv("PETAB_MCP_BUILD_ROOT", raising=False)
    monkeypatch.setattr(cli, "_send_stderr_to", _never)
    assert cli.main(["--petab-yaml", "problem.yaml", *flags]) == 2
    assert says in capsys.readouterr().err


class _Stop(Exception):
    """Whatever ends the server: an unreadable problem in ``create_server``, Ctrl-C in ``run``."""


def _fake_server(monkeypatch, events, fails_in=None):
    """Replace ``create_server`` with one that records the ``cluster`` it is given, and whose server
    records its ``run``; ``fails_in`` names which of the two then raises :class:`_Stop`. ``main``
    also leaves this process's fd 1 and fd 2 alone, since pytest is capturing them."""

    class Server:
        def run(self, transport):
            events.append("run")
            if fails_in == "run":
                raise _Stop

    def create_server(*args, cluster, **kwargs):
        events.append(("create_server", cluster))
        if fails_in == "create_server":
            raise _Stop
        return Server()

    monkeypatch.setattr(cli, "create_server", create_server)
    monkeypatch.setattr(cli, "_send_stderr_to", lambda path: None)
    monkeypatch.setattr(cli, "_reserve_stdout_for_the_protocol", lambda: None)


@pytest.mark.parametrize("fails_in", ["create_server", "run"])
def test_the_server_gets_the_pool_and_takes_it_down_when_it_stops(monkeypatch, tmp_path, fails_in):
    """``main`` builds ``SLURMCluster()`` with no arguments, bounds it in jobs by the two flags,
    hands it to ``create_server``, and closes it (cancelling its jobs) however the server ends, even
    when it never starts. The cluster is faked, so no scheduler starts."""
    events, pools = [], []

    class Cluster:
        def __init__(self, *args, **kwargs):
            pools.append(self)
            events.append(("SLURMCluster", args, kwargs))

        def adapt(self, **bounds):
            events.append(("adapt", bounds))

        def close(self):
            events.append("close")

    monkeypatch.setattr(dask_jobqueue, "SLURMCluster", Cluster)
    _fake_server(monkeypatch, events, fails_in)
    argv = ["--petab-yaml", "problem.yaml", "--build-root", str(tmp_path)]
    with pytest.raises(_Stop):
        cli.main([*argv, "--slurm-max-jobs", "4", "--slurm-min-jobs", "1"])
    ran = ["run"] if fails_in == "run" else []
    assert events == [
        ("SLURMCluster", (), {}),
        ("adapt", {"minimum_jobs": 1, "maximum_jobs": 4}),
        ("create_server", pools[0]),
        *ran,
        "close",
    ]


def test_without_slurm_max_jobs_the_cli_needs_no_dask(monkeypatch):
    """The pool is opt-in: ``create_server`` gets ``cluster=None``. Neither importing the CLI nor
    running it imports ``dask_jobqueue``, so it runs without the ``dask`` extra (which CI always
    installs)."""
    monkeypatch.setitem(sys.modules, "dask_jobqueue", None)  # importing it now fails
    importlib.reload(cli)
    events = []
    _fake_server(monkeypatch, events)
    assert cli.main(["--petab-yaml", "problem.yaml"]) == 0
    assert events == [("create_server", None), "run"]
