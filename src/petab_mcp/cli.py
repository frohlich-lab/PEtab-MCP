"""Command-line entry point: ``petab-mcp --petab-yaml PATH``.

Starts the local MCP server (stdio transport) for the given PEtab v1 problem.
"""

from __future__ import annotations

import argparse
import os
import sys

from . import criteria
from .server import create_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="petab-mcp",
        description="Local MCP server for interacting with a PEtab v1 problem.",
    )
    parser.add_argument(
        "--petab-yaml",
        default=os.environ.get("PETAB_MCP_YAML"),
        help="Path to the PEtab v1 problem YAML (or set PETAB_MCP_YAML).",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.petab_yaml:
        print(
            "error: no PEtab YAML given. Pass --petab-yaml PATH or set PETAB_MCP_YAML.",
            file=sys.stderr,
        )
        return 2
    server = create_server(
        args.petab_yaml,
        build_root=args.build_root,
        provenance_dir=args.provenance_dir,
        reference_mode=args.reference_validation,
        skills_dir=args.skills_dir,
        criterion=args.criterion,
    )
    server.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
