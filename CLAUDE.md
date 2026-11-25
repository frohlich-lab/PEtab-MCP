# Working in this repo

**`petab_mcp`** (under `src/`) is the MCP server. It wraps one PEtab v1 parameter-estimation
problem and exposes a _deliberately constrained_ editing loop. The constraints are the product,
not scaffolding.

The agent harness that drives it (panel, judge, benchmark sweep, run explorer) lives in
[frohlich-lab/petab_mcp](https://github.com/frohlich-lab/petab_mcp) and installs this package
pinned to a commit; a change here reaches it only when that pin is bumped.

## Read before you change behaviour

The "Overview" section of [README.md](README.md) and [docs/DESIGN.md](docs/DESIGN.md). The
design principles of the package are defined there.

## Writing code here

Refer to [docs/STYLE_GUIDE.md](docs/STYLE_GUIDE.md).
