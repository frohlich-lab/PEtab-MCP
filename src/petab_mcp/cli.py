"""Command-line entry point: ``petab-mcp --petab-yaml PATH``.

Starts the local MCP server (stdio transport) for the given PEtab v2 problem.
"""

from __future__ import annotations

import argparse
import os
import sys

from .server import create_server


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="petab-mcp",
        description="Local MCP server for interacting with a PEtab v2 problem.",
    )
    parser.add_argument(
        "--petab-yaml",
        default=os.environ.get("PETAB_MCP_YAML"),
        help="Path to the PEtab v2 problem YAML (or set PETAB_MCP_YAML).",
    )
    parser.add_argument(
        "--build-root",
        default=os.environ.get("PETAB_MCP_BUILD_ROOT"),
        help="Directory for AMICI-compiled models (defaults to a temp dir).",
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
    server = create_server(args.petab_yaml, build_root=args.build_root)
    server.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
