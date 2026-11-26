"""``petab-mcp-viz``: serve the viewer for one problem's store."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="petab-mcp-viz",
        description="A read-only web viewer of a petab-mcp session's version store.",
    )
    parser.add_argument("--petab-yaml", required=True, help="the problem the server was given")
    parser.add_argument("--provenance-dir", help="the store, if not <problem dir>/.petab_mcp")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument("--address", default="localhost")
    parser.add_argument(
        "--poll",
        type=float,
        default=3.0,
        help="seconds between looks at the store for a new version (0: never)",
    )
    parser.add_argument("--no-show", action="store_true", help="do not open a browser")
    args = parser.parse_args(argv)
    try:
        import panel as pn

        from .app import build
    except ImportError as exc:
        sys.exit(f"petab-mcp-viz needs the viz extra (pip install 'petab-mcp[viz]'): {exc}")
    from ..session import store_root

    root = store_root(args.petab_yaml, args.provenance_dir)
    yaml = Path(args.petab_yaml)
    pn.serve(
        {"/": lambda: build(root, yaml, args.poll)},
        port=args.port,
        address=args.address,
        show=not args.no_show,
        title=f"PEtab-MCP · {yaml.stem}",
    )


if __name__ == "__main__":
    main()
