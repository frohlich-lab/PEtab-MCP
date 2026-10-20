# Coding standards

How code is written here. This is descriptive: it was derived from the existing code, so
matching it means matching the code around you. For _what the system is_, see the
[README.md](../README.md).

## Rationale log

A non-obvious implementation choice is prepended to [RATIONALE.md](RATIONALE.md) and
references the source code where the implementation lives. The explanation names **the
failure that motivated it** — with the issue or PR number, the sweep date, or the measured
cost, whichever exists.

**Writing one is the exception.** Most PRs add nothing; the budget when they do is one entry
of two or three sentences, and a "why" that belongs to one module that still exists goes in that
module's docstring instead. See the header of [RATIONALE.md](RATIONALE.md) before adding one.

Two rules follow:

- **Negative knowledge is load-bearing.** "Deliberately NOT x", "do not add this back", "the
  first version of this matched … which caught …" — these are the record of what was already
  tried.
- **Write one when you make a call a reader would otherwise undo.** If the obvious
  simplification is wrong, the rationale log is the only thing standing between it and the
  next contributor.

This log explains _why_. It does not restate what the source code does.

## Docstrings

- **Every module opens with a substantial docstring**: what it is, what it owns, and why it
  is a separate module. For example, [messages.py](../src/petab_mcp/messages.py).
- **Prose, not numpydoc.** `Parameters ----------` blocks appear about once in
  `src/petab_mcp` and are reserved for structured exception constructors
  ([errors.py](../src/petab_mcp/errors.py)). Everywhere else, explain in sentences.
- **Sphinx roles and double backticks**: `:func:`, `:mod:`, `:class:`, ` ``literal`` `.
- **`#:` for documented module-level constants**, so the comment attaches to the name.
- **Tool docstrings are a contract with a model, not with a human.** They are what the agent
  reads, they are the canonical statement of their own arguments. Edit them with the care of
  changing an API signature.
- **One tool, one description.** A tool registered under two schemas (an ablation arm) gets ONE
  composed description, not a docstring per registration — see `_make_changes_doc` in
  [server.py](../src/petab_mcp/server.py).

## Agent-facing prose

A tool docstring is what every client's agent reads on every turn, the harness's panel
included: the tool's contract and what an agent needs to use it, and nothing a client's own job
adds (its stopping rule, its other agents). The longer reference (every result field, worked SQL)
is a skill, `_skills/reading-fits`, which the docstring points at by name.

- **No argument lists.** The JSON schema is the signature and is sent alongside the description,
  so `Args: a, b, c` is a hand-maintained copy of it. Say what an argument MEANS when the name
  does not; never enumerate them.

## Types

- **`from __future__ import annotations` at the top of every module** — every module in
  `src/` and every test file.
- Builtin generics and `X | None`; `requires-python = ">=3.12"`, so there is no reason to
  reach for `typing.List` or `Optional`.
- **Return annotations on public functions** (~457 of 528 `def`s). Private one-liners may
  skip.

## Errors

Everything surfaced to the client derives from `PetabMcpError`
([errors.py](../src/petab_mcp/errors.py)). Never raise a bare `ValueError` on a path that
reaches a tool — the point of the typed hierarchy is that the tool boundary can tell "the
caller asked for something impossible" from "we have a bug", and turn the first into an
actionable message.

## Where things live

- **[constants.py](../src/petab_mcp/constants.py)** — cross-module numeric knobs, each with
  the reasoning for its value. Its own docstring lists what deliberately stays module-local
  (the SBML lint categories, the AMICI hints, the SQL schema, the E-utilities URLs, the
  fitter's env-var knobs).
- **[messages.py](../src/petab_mcp/messages.py)** — the diagnostic stack a `make_changes`
  result carries. One ordered list, each entry tagged `level` and `source`, in pipeline
  order.
- **[errors.py](../src/petab_mcp/errors.py)** — the exception types.
- Tools stay thin. [server.py](../src/petab_mcp/server.py) wires and validates; the work
  lives in `session` / `operations` / `fitting` / `data_access` / `reconcile`.

## Dependencies

- **Every pin carries a why-comment** naming what breaks outside the range. `fastmcp>=3,<4`
  documents the whole mcp-2.0 / langchain-mcp-adapters chain that forces the upper bound.
- **An upper bound needs a "revisit once…" condition**, so the next person knows what would
  let them lift it.
- **A new import means a new pin.** Several dependencies are pinned precisely because
  nothing imports them directly (`h5py` is written _through_ pyPESTO's storage API) — a
  transitive graph change would otherwise break them silently.
- **Optional deps must not silently skip tests.** Adding an `importorskip` means adding its
  dependency to `dev`.

## Tests

- `tests/test_<module>.py`; plain `pytest` functions, no class wrappers.
- Short aliased imports of the module under test: `from petab_mcp import operations as O`.
- Module-level `_BASE_*` template strings with a `{extra}` splice point, plus a small
  `_helper` that fills it, rather than a fixture per variation
  ([test_operations.py](../tests/test_operations.py)).
- **A test comment names the regression it locks in.** A test whose failure does not tell
  you what broke is half a test — see the docstrings in
  [test_noise_sigma_advice.py](../tests/test_noise_sigma_advice.py).
- **Do not guard prose against prose.** A regex asserting that N paraphrases of one rule still
  agree is a maintenance cost standing in for a structural fix: make it one string and interpolate
  it.
- **`pytest` takes no `-n` from config.** The caller passes it; CI passes `-n 2`.

## Linting

`ruff check .` and `ruff format .` — config and the reason for every entry in `[tool.ruff]`
in [pyproject.toml](../pyproject.toml). CI runs `ruff check` as its own fast job (ruff only,
no toolchain).

The rule set is `E`, `F`, `W`, minus `E501`. It is narrow on purpose: it was
configured onto a codebase that had never run a linter, so it is what the code already
satisfies.

When a ruff upgrade lands a rule that is wrong for this repo, add it to `ignore` **with the
reason** — do not cap ruff.

## Commits

`area: what changed (#PR)` — lowercase area, imperative, matching `git log`. The area is a
module or subsystem (`fit:`, `server:`, `validation:`, `reconcile:`, `deps:`) and may be
omitted for a change that spans the repo.
