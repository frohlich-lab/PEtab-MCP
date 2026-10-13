"""Command-line entry point: ``petab-mcp --petab-yaml PATH``.

Starts the local MCP server (stdio transport) for the given PEtab problem (v2; a v1 problem
is upgraded on load). The process's stderr goes to ``<store>/stderr.log``. With
``--slurm-max-jobs N`` the fits run on a pool of up to N SLURM jobs that the server submits and
cancels (README, "Fitting on SLURM").
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any, BinaryIO

from . import criteria
from .server import create_server
from .session import store_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="petab-mcp",
        description="Local MCP server for interacting with a PEtab v2 problem.",
    )
    parser.add_argument(
        "--petab-yaml",
        default=os.environ.get("PETAB_MCP_YAML"),
        help="Path to the PEtab problem YAML, v2 or v1 (or set PETAB_MCP_YAML).",
    )
    parser.add_argument(
        "--build-root",
        default=os.environ.get("PETAB_MCP_BUILD_ROOT"),
        help="Directory for AMICI-compiled models (defaults to a temp dir).",
    )
    parser.add_argument(
        "--provenance-dir",
        default=os.environ.get("PETAB_MCP_PROVENANCE_DIR"),
        help="Directory for provenance + version snapshots (default: <problem>/.petab_mcp).",
    )
    parser.add_argument(
        "--reference-validation",
        default=os.environ.get("PETAB_MCP_REFERENCE_VALIDATION", "best_effort"),
        choices=["required", "best_effort", "off"],
        help="How strictly to validate operation references against PubMed.",
    )
    parser.add_argument(
        "--skills-dir",
        default=None,
        help="Directory of tool-usage skills to expose "
        "(default: PETAB_MCP_SKILLS_DIR, else the bundled _skills).",
    )
    parser.add_argument(
        "--criterion",
        # `None`, NOT the environment's value: the session resolves an omitted criterion
        # (argument, else environment, else bic — `criteria.resolve`) in the one place that holds
        # it, so there is a single read of the variable rather than several that could disagree.
        default=None,
        choices=sorted(criteria.PENALTIES),
        help="Complexity penalty to rank models by. bic = k*ln(n) (the default); "
        "caic = k*(ln(n)+1), stricter still; aic = 2k; aicc = AIC with the small-sample "
        "correction; hqc = Hannan-Quinn, 2k*ln(ln(n)); tic = Takeuchi, 2*k_eff, charging the "
        "effective parameter count the fit reports. Defaults to PETAB_MCP_CRITERION, else bic.",
    )
    parser.add_argument(
        "--transport",
        default="stdio",
        choices=["stdio", "sse", "streamable-http"],
        help="MCP transport (default: stdio).",
    )
    parser.add_argument(
        "--slurm-max-jobs",
        type=int,
        default=0,
        metavar="N",
        help="Run fits on a pool of at most N SLURM jobs that this server submits and cancels "
        "(default 0: on this node). The jobs are set up in dask's jobqueue.slurm config; needs "
        "the dask extra and a --build-root the compute nodes share.",
    )
    parser.add_argument(
        "--slurm-min-jobs",
        type=int,
        default=0,
        metavar="M",
        help="SLURM jobs the pool keeps while idle, at most N (default 0).",
    )
    return parser


def _send_stderr_to(path: Path) -> None:
    """Point fd 2 at ``path``, because nobody reads a server's stderr. This catches Python, every
    logging stream handler (AMICI's rate-limited one, fides'), native code, compilers and the
    pyPESTO workers, which inherit fd 2. One line on the real stderr says where it all went."""
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"petab-mcp: stderr goes to {path}", file=sys.stderr, flush=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(fd, 2)
    os.close(fd)


class _ProtocolStdout:
    """``sys.stdout`` under the stdio transport: ``buffer`` is the JSON-RPC channel, and any text
    written goes to stderr."""

    def __init__(self, channel: BinaryIO) -> None:
        self.buffer = channel

    def __getattr__(self, name: str) -> Any:
        return getattr(sys.stderr, name)


def _reserve_stdout_for_the_protocol() -> None:
    """Under the stdio transport fd 1 IS the JSON-RPC channel, and neither mcp nor FastMCP guards it:
    AMICI ``print()``s a failed build's whole compiler log to stdout, and native code, compiler
    subprocesses and pyPESTO workers inherit fd 1. In the channel, any of it is a line the client
    cannot parse, or a partial line that swallows the reply written after it. So the protocol gets
    a private duplicate of the channel and fd 1 becomes stderr for everything else. The MCP SDK
    wraps ``sys.stdout.buffer`` once, when the transport starts, so that is the one attribute that
    has to lead to the channel."""
    channel = os.fdopen(os.dup(1), "wb")
    os.dup2(2, 1)
    sys.stdout = _ProtocolStdout(channel)


def _slurm_pool(min_jobs: int, max_jobs: int) -> Any:
    """A ``dask_jobqueue.SLURMCluster`` that grows to ``max_jobs`` jobs while fits wait for workers
    and shrinks to ``min_jobs`` when idle. It is given no SLURM settings here: partition, cores,
    processes, memory, walltime, worker lifetime and directories all come from dask's own
    ``jobqueue.slurm`` config."""
    from dask_jobqueue import SLURMCluster  # the optional `dask` extra

    cluster = SLURMCluster()
    cluster.adapt(minimum_jobs=min_jobs, maximum_jobs=max_jobs)
    return cluster


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.petab_yaml:
        print(
            "error: no PEtab YAML given. Pass --petab-yaml PATH or set PETAB_MCP_YAML.",
            file=sys.stderr,
        )
        return 2
    if not 0 <= args.slurm_min_jobs <= args.slurm_max_jobs:
        print(
            "error: the SLURM pool needs 0 <= --slurm-min-jobs <= --slurm-max-jobs.",
            file=sys.stderr,
        )
        return 2
    if args.slurm_max_jobs > 0 and not args.build_root:
        print(
            "error: --slurm-max-jobs needs --build-root (or PETAB_MCP_BUILD_ROOT) on storage the "
            "compute nodes share: the workers import the compiled models by the same path, and "
            "the default temp dir is local to this node.",
            file=sys.stderr,
        )
        return 2
    # stderr first: the stdout guard points fd 1 at whatever fd 2 is by then.
    _send_stderr_to(store_root(args.petab_yaml, args.provenance_dir) / "stderr.log")
    if args.transport == "stdio":
        _reserve_stdout_for_the_protocol()
    cluster = None
    if args.slurm_max_jobs > 0:
        cluster = _slurm_pool(args.slurm_min_jobs, args.slurm_max_jobs)
    try:
        server = create_server(
            args.petab_yaml,
            build_root=args.build_root,
            provenance_dir=args.provenance_dir,
            reference_mode=args.reference_validation,
            skills_dir=args.skills_dir,
            criterion=args.criterion,
            cluster=cluster,
        )
        server.run(transport=args.transport)
    finally:
        if cluster is not None:
            cluster.close()  # cancels its SLURM jobs
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
