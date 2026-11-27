"""Provenance and versioning, persisted to disk.

Under ``<store_root>/`` (default ``<problem_dir>/.petab_mcp``):

* ``provenance.jsonl`` — append-only event log (edits, fits, rollbacks, resets). Each
  event carries a UTC timestamp.
* ``versions/vN/`` — a snapshot of the mutable model state at version N: ``model.ant``
  (the canonical Antimony) and ``parameters.tsv`` (the parameter table, as a PEtab v2 TSV; its
  estimated rows' nominal values are the one place a fit's values are kept by name), plus
  ``meta.json`` (label, kind, parent, timestamp, scores, and ``edit_warnings``: the warnings
  its change and lint gave, whole). A fitted version also carries
  ``simulations.json`` (the measured-vs-simulated dataframe + residuals), ``optimization.json``
  (the pyPESTO per-start + per-iteration trace records), and ``pypesto_result.hdf5`` (the
  pyPESTO result file, copied in from the per-model cache) — so rolling back restores a
  version's fit artifacts, not just its model. A model built from statements also carries
  ``statements.json``, its rows and their bindings (read through :func:`statements_record`). A
  version scored on held-out data also carries :data:`VALIDATION_SIMULATIONS`, which no
  agent-facing read opens.
* ``cache/`` — pyPESTO result files keyed by problem hash (written on each fit, copied into
  version dirs when snapshotted).
* ``HEAD`` — the current version number.
* :data:`FIT_PROGRESS` — the running change's fit, for a viewer in another process.
* ``FORMAT`` — :data:`STORE_FORMAT`. A store without it was written by the PEtab v1 server, whose
  versions hold v1 parameter tables and condition-keyed simulation records; it is refused rather
  than read, since nothing here can tell which of its numbers still mean what they meant.

The store is pure storage: it records Antimony text + parameter records and reads them
back. Applying a version to the live problem lives in the session/operations layer.
"""

from __future__ import annotations

import difflib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from collections.abc import Iterable
from typing import Any


#: The layout a store's ``FORMAT`` file names (see the module docstring).
STORE_FORMAT = "petab-v2"

#: The running change's fit as :class:`multistart.Progress` publishes it, there only while a change
#: runs: nothing in the server reads it back.
FIT_PROGRESS = "fit_progress.json"

#: What :meth:`ProvenanceStore.get_version` can add to a version's meta, model and parameters: the
#: fit's per-point records (``simulations.json``) and its starts with their traces
#: (``optimization.json``), each far larger than the rest.
VERSION_PARTS = ("simulation", "optimization")

#: The fields of a version's ``meta.json`` that ``list_versions`` gives one row each; the rest of
#: the meta is the version's own read.
SUMMARY_FIELDS = (
    "version",
    "parent",
    "neg_log_likelihood",
    "criterion_value",
    "k",
    "is_new_best",
    "residual_flagged",
)

#: A version's held-out score with its per-point records
#: (:func:`petab_mcp.validation_data.score_session`); a harness that archives a version's files
#: names it too.
VALIDATION_SIMULATIONS = "validation_simulations.json"


def model_diff(before: str, after: str, fromfile: str = "", tofile: str = "") -> str:
    """The unified diff of two Antimony sources: ``diff_versions``' ``model_diff``, and what
    :mod:`nesting` reads a commit's removed and rewritten reactions and rules from."""
    return "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=fromfile,
            tofile=tofile,
        )
    )


def binding_of(row: dict[str, Any]) -> dict[str, Any]:
    """A row's binding as a statements record keeps it, ``{policy, parameters?}``, from a row that
    carries its ``policy`` and ``parameters``."""
    return {"policy": row["policy"]} | (
        {"parameters": dict(row["parameters"])} if row.get("parameters") else {}
    )


def statements_record(record: dict[str, Any] | None) -> dict[str, Any] | None:
    """A stored statements record (``statements.json``) as the server reads it now: its ``rows``
    each the statement, and its ``bindings`` each row's policy and parameter bindings by prefix
    (:func:`operations.statements_to_antimony`). A record stored before the two were given apart
    kept each row's ``policy`` and ``parameters`` on the row, and is read as the same rows and
    bindings, so a resume, a ``base_version`` fork or ``get_model`` of an older version reads it
    as it was."""
    if not record or "bindings" in record:
        return record
    split = {"policy", "parameters"}
    return {
        **record,
        "rows": [
            {k: v for k, v in row.items() if k not in split} for row in record.get("rows", [])
        ],
        "bindings": {row["prefix"]: binding_of(row) for row in record.get("rows", [])},
    }


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def summary(meta: dict[str, Any]) -> dict[str, Any]:
    """``meta``'s :data:`SUMMARY_FIELDS`, ``None`` where it records none."""
    return {key: meta.get(key) for key in SUMMARY_FIELDS}


def _dumps(obj: Any) -> str:
    """``obj`` as compact JSON: programs read the store, not people."""
    return json.dumps(obj, separators=(",", ":"))


class ProvenanceStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.log_path = self.root / "provenance.jsonl"
        self.errors_path = self.root / "errors.jsonl"
        self.head_path = self.root / "HEAD"
        format_path = self.root / "FORMAT"
        if self.versions_dir.is_dir() and any(self.versions_dir.iterdir()):
            found = format_path.read_text().strip() if format_path.is_file() else None
            if found != STORE_FORMAT:
                from .errors import PetabMcpError

                raise PetabMcpError(
                    f"the provenance store at {self.root} is in format {found or 'petab-v1'!r}, "
                    f"not {STORE_FORMAT!r}: it was written by an earlier server and cannot be "
                    "resumed. Start the server with a fresh --provenance-dir."
                )
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        format_path.write_text(STORE_FORMAT)

    # -- event log ---------------------------------------------------------
    def append_event(self, event: dict[str, Any]) -> dict[str, Any]:
        event = {"timestamp": _now(), **event}
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(_dumps(event) + "\n")
        return event

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._tail(self.log_path, limit)

    # -- error log (separate from the success-oriented provenance log) ------
    def log_error(self, record: dict[str, Any]) -> dict[str, Any]:
        record = {"timestamp": _now(), **record}
        with self.errors_path.open("a", encoding="utf-8") as fh:
            fh.write(_dumps(record) + "\n")
        return record

    def recent_errors(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._tail(self.errors_path, limit)

    @staticmethod
    def _tail(path: Path, limit: int) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        with path.open(encoding="utf-8") as fh:
            records = [json.loads(ln) for ln in fh if ln.strip()]
        return records[-limit:] if limit and limit > 0 else records

    # -- HEAD --------------------------------------------------------------
    @property
    def current_version(self) -> int:
        if not self.head_path.exists():
            return -1
        try:
            return int(self.head_path.read_text().strip())
        except ValueError:
            return -1

    def _set_head(self, n: int) -> None:
        self.head_path.write_text(str(n))

    @property
    def last_version(self) -> int:
        """The highest version number on disk, or -1. HEAD can point at an EARLIER version — a
        restore moves it back — so this, not HEAD, is what the next snapshot counts from."""
        vdirs = [
            int(d.name[1:])
            for d in self.versions_dir.glob("v*")
            if d.is_dir() and d.name[1:].isdigit()
        ]
        return max(vdirs, default=-1)

    def set_head(self, n: int) -> None:
        """Point HEAD at an existing version without appending a new one."""
        self._version_dir(n)  # raises UnknownEntityError if absent
        self._set_head(n)

    # -- versions ----------------------------------------------------------
    def snapshot_version(
        self,
        model_ant: str,
        parameters: Any,
        label: str | None = None,
        kind: str = "snapshot",
        parent: int | None = None,
        model_source: str | None = None,
        statements: dict[str, Any] | None = None,
        meta: dict[str, Any] | None = None,
        simulation: dict[str, Any] | None = None,
        optimization: dict[str, Any] | None = None,
    ) -> int:
        n = self.last_version + 1
        vdir = self.versions_dir / f"v{n}"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "model.ant").write_text(model_ant, encoding="utf-8")
        # The Antimony the caller SUBMITTED, next to the build output: `model.ant` is what the
        # reconcile and libantimony produced from it, and what the next edit applies to. This is
        # a record of what went in, which nothing reads back; a commit that submits no model text
        # has none.
        if model_source:
            (vdir / "model_source.ant").write_text(model_source, encoding="utf-8")
        # What a `statements` commit asserted, beside the source it generated: the row is what
        # a later `revise` addresses, and it cannot be recovered from the expanded model. A
        # commit that submits no model text inherits its rows (see session._snapshot_version).
        if statements:
            (vdir / "statements.json").write_text(_dumps(statements), encoding="utf-8")
        # The parameter table is stored as a DataFrame (PEtab-style TSV), not JSON records,
        # so it round-trips as a real table with its dtypes intact.
        from .parameters import parameter_df_to_tsv

        (vdir / "parameters.tsv").write_text(parameter_df_to_tsv(parameters), encoding="utf-8")
        meta_obj = {
            "version": n,
            "label": label,
            "kind": kind,
            "parent": self.current_version if parent is None else parent,
            "timestamp": _now(),
            **(meta or {}),
        }
        (vdir / "meta.json").write_text(_dumps(meta_obj), encoding="utf-8")
        # The simulation dataframe generated for this version (measured vs simulated +
        # residuals), read back by get_version.
        if simulation is not None:
            (vdir / "simulations.json").write_text(_dumps(simulation), encoding="utf-8")
        # The optimization traces/results for this version: the per-start + per-iteration
        # records (read back into the DuckDB tables) as JSON, plus the pyPESTO HDF5 result
        # file copied in from the per-model cache (or the source version, on rollback).
        if optimization is not None:
            self._write_optimization(vdir, optimization)
        self._set_head(n)
        return n

    @staticmethod
    def _write_optimization(vdir: Path, optimization: dict[str, Any]) -> None:
        persist = {
            k: optimization.get(k)
            # x_names/x_scales label each start's positional `x` vector; without them a
            # restored version's per-start optima cannot be read back (see
            # fitting.previous_optima, which warm-starts the next fit from them).
            # exitflag_counts is the ONLY surviving record of how each start ended --
            # the per-start `exitflag`/`message` columns were removed in favour of it and
            # the optimizer log runs at ERROR -- so a restored version cannot answer
            # "optimiser-limited or structure-limited?" without it. This whitelist
            # silently dropped it: a key absent here is not persisted and nothing warns.
            # n_warm_starts says which starts were warm (pyPESTO numbers them first), for
            # petab_mcp.viz; it too was dropped here.
            for k in (
                "n_starts",
                "starts",
                "trace",
                "x_names",
                "x_scales",
                "exitflag_counts",
                "n_starts_killed",
                "n_warm_starts",
            )
        }
        (vdir / "optimization.json").write_text(_dumps(persist), encoding="utf-8")
        result_file = optimization.get("result_file")
        if result_file:
            try:
                src = Path(result_file)
                if src.is_file():
                    shutil.copyfile(src, vdir / "pypesto_result.hdf5")
            except OSError:  # copying the interop hdf5 must never fail a snapshot
                pass

    def write_validation_simulation(self, n: int, payload: dict[str, Any]) -> Path:
        """Write version ``n``'s held-out ``payload`` (:data:`VALIDATION_SIMULATIONS`), replacing
        an earlier score of it; returns the file."""
        path = self._version_dir(n) / VALIDATION_SIMULATIONS
        path.write_text(_dumps(payload), encoding="utf-8")
        return path

    def _version_dir(self, n: int) -> Path:
        vdir = self.versions_dir / f"v{n}"
        if not vdir.is_dir():
            from .errors import UnknownEntityError

            raise UnknownEntityError(f"No such version: v{n}")
        return vdir

    def get_meta(self, n: int) -> dict[str, Any]:
        """Read a version's ``meta.json`` ONLY (label/kind/fit metrics/structure_hash/fit_status).

        Cheap: it does not touch the model, parameter table, simulation records, or optimization
        trace (unlike :meth:`get_version`), so callers that only need meta fields — e.g. the fit
        de-dup check — avoid a large disk read on the hot path.
        """
        return json.loads((self._version_dir(n) / "meta.json").read_text())

    def get_parameter_df(self, n: int):
        """Read a version's parameter table back as a DataFrame (index = ``parameterId``)."""
        return self._read_parameter_df(self._version_dir(n))

    def get_statements(self, n: int) -> dict[str, Any] | None:
        """A version's statements record (``statements.json``), or None for a model not built
        from statements. Raises ``UnknownEntityError`` for a version that does not exist."""
        f = self._version_dir(n) / "statements.json"
        return statements_record(json.loads(f.read_text(encoding="utf-8"))) if f.is_file() else None

    def get_model_antimony(self, n: int) -> str:
        """Read a version's ``model.ant`` ONLY — cheap, unlike :meth:`get_version`, which also
        pulls the (large) per-point simulation and optimization blobs."""
        return (self._version_dir(n) / "model.ant").read_text(encoding="utf-8")

    @staticmethod
    def _read_parameter_df(vdir: Path):
        """Load ``parameters.tsv`` as a DataFrame in :mod:`petab_mcp.problem`'s conventions."""
        import pandas as pd

        from .problem import normalize_parameter_df

        return normalize_parameter_df(pd.read_csv(vdir / "parameters.tsv", sep="\t", index_col=0))

    def get_version_simulation(self, n: int) -> "dict[str, Any] | None":
        """Version ``n``'s stored simulation blob (``simulations.json``), or ``None``.

        A LIGHT read for assembling the ``simulations_history`` SQL table: unlike
        :meth:`get_version` it loads neither the model text, the parameter table, nor the
        optimization payload — history assembly touches every version, so it must not pay for
        the four other files per version.

        Reached by :meth:`stored_version_numbers` + this method, never by ``list_versions``:
        the history must not depend on ``meta.json``'s health (a version with an intact
        simulation would otherwise vanish from every comparison because its unrelated meta was
        truncated — the very interrupted-write case this is hardened against).

        ``None`` covers every way the blob can be missing or unusable: an absent version dir,
        an absent file (a failed/skipped fit stores no simulation), an unreadable or corrupt
        one (a snapshot interrupted mid-write — wall-clock kill, disk full — must degrade this
        read, not poison every ``query_data``/``list_data_tables`` call for the rest of the
        session), and a file that parses to a non-object. That is :meth:`list_versions`'s
        per-file defensiveness, NOT :meth:`get_statements`'s raise-on-bad-argument contract:
        history assembly iterates versions it did not choose, so a bad one is skippable data
        here, never a caller error.
        """
        p = self._version_path(n) / "simulations.json"
        try:
            obj = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            if p.is_file():  # exists but unreadable/corrupt — worth a line; absence is normal
                logging.getLogger("petab_mcp").warning(
                    "version v%s has an unreadable simulations.json; it contributes no rows "
                    "to simulations_history",
                    n,
                    exc_info=True,
                )
            return None
        if not isinstance(obj, dict):
            logging.getLogger("petab_mcp").warning(
                "version v%s simulations.json is not a JSON object (%s); it contributes no "
                "rows to simulations_history",
                n,
                type(obj).__name__,
            )
            return None
        return obj

    def simulation_fingerprint(self, n: int) -> "tuple[int, int] | None":
        """``(size, mtime_ns)`` of version ``n``'s ``simulations.json``, or ``None`` if absent.

        A stat, not a parse — the cache-invalidation key for ``simulations_history``. Keying
        on the FILE rather than on the version list is what closes the mid-commit window:
        ``snapshot_version`` writes ``meta.json`` before ``simulations.json``, so a reader can
        see version N listed while its simulation does not exist yet; a version-tuple key
        would cache that empty view permanently, whereas this key changes the moment the file
        lands.
        """
        try:
            st = (self._version_path(n) / "simulations.json").stat()
        except OSError:
            return None
        return (st.st_size, st.st_mtime_ns)

    def _version_path(self, n: int) -> Path:
        """``versions/v<n>`` without existence checking — the NON-raising sibling of
        :meth:`_version_dir`.

        The two light reads above iterate versions they did not choose, so a missing dir is
        skippable data, not the bad argument ``_version_dir`` rightly raises on. ``int(n)``
        keeps the path canonical: a ``meta.json`` carrying ``"version": 3.0`` (callers can
        splat their own ``meta`` over the field) would otherwise format as ``v3.0``, miss
        silently, and drop that version's rows with no warning.
        """
        return self.versions_dir / f"v{int(n)}"

    def stored_version_numbers(self) -> "list[int]":
        """Version numbers present on disk, ascending — from the DIRECTORY NAMES.

        Deliberately not ``list_versions``: that reads and JSON-parses every ``meta.json``
        (waste when only the numbers are wanted) and SKIPS any version whose meta is
        unreadable, which would silently drop an intact simulation from the history. Same
        ``v<int>`` source :attr:`last_version` uses, so a stray ``v*`` entry cannot raise here.
        """
        return sorted(
            int(d.name[1:])
            for d in self.versions_dir.glob("v*")
            if d.is_dir() and d.name[1:].isdigit()
        )

    def get_version(self, n: int, include: Iterable[str] = VERSION_PARTS) -> dict[str, Any]:
        """Version ``n``'s meta, ``model_antimony`` and ``parameters``, plus each of
        :data:`VERSION_PARTS` in ``include`` (``None`` where the version stored none); a part not
        in ``include`` is neither read nor returned."""
        vdir = self._version_dir(n)
        meta = json.loads((vdir / "meta.json").read_text())
        parts = {}
        if "simulation" in include:
            sim_path = vdir / "simulations.json"
            parts["simulation"] = json.loads(sim_path.read_text()) if sim_path.is_file() else None
        if "optimization" in include:
            opt_path = vdir / "optimization.json"
            optimization = json.loads(opt_path.read_text()) if opt_path.is_file() else None
            if optimization is not None:
                # Point at this version's copy of the pyPESTO result file (if it was stored), so a
                # later rollback can copy it forward and tools can locate it.
                hdf5 = vdir / "pypesto_result.hdf5"
                optimization["result_file"] = str(hdf5) if hdf5.is_file() else None
            parts["optimization"] = optimization
        # `parameters` stays a list of JSON records — the shape the MCP tool and diff want
        # (and it must be JSON-serializable). The raw DataFrame is available separately via
        # `get_parameter_df`, which the session uses to restore state directly.
        from .parameters import parameter_df_to_records

        return {
            **meta,
            "model_antimony": (vdir / "model.ant").read_text(encoding="utf-8"),
            "parameters": parameter_df_to_records(self._read_parameter_df(vdir)),
            **parts,
        }

    def list_versions(self, since: int | None = None) -> list[dict[str, Any]]:
        """Every version's stored ``meta.json``, oldest first; with ``since``, only the versions
        after it.

        An entry that is unreadable, malformed, or not a JSON OBJECT is skipped and logged, not
        returned: callers iterate the result and call ``.get`` on each element without guarding,
        and a raise there escapes into the commit path. Callers also degrade a raise from this
        method to an empty list, which would silently disable de-duplication. Skipping per file
        degrades proportionally.
        """
        out = []
        # Only ``v<int>`` directories are versions; skip any stray ``v*`` entry (a leftover temp
        # dir, an editor artefact) so `int(name[1:])` can't raise and break version listing.
        vdirs = [d for d in self.versions_dir.glob("v*") if d.is_dir() and d.name[1:].isdigit()]
        for vdir in sorted(vdirs, key=lambda p: int(p.name[1:])):
            if since is not None and int(vdir.name[1:]) <= since:
                continue
            try:
                meta = json.loads((vdir / "meta.json").read_text())
            except (OSError, ValueError):
                logging.getLogger("petab_mcp").warning(
                    "version %s has an unreadable meta.json; excluded from the version listing",
                    vdir.name,
                    exc_info=True,
                )
                continue
            if not isinstance(meta, dict):
                logging.getLogger("petab_mcp").warning(
                    "version %s has a meta.json that is not a JSON object (%s); excluded from "
                    "the version listing",
                    vdir.name,
                    type(meta).__name__,
                )
                continue
            out.append(meta)
        return out

    def diff_versions(self, a: int, b: int) -> dict[str, Any]:
        va, vb = self.get_version(a, include=()), self.get_version(b, include=())
        pa = {p["parameterId"]: p for p in va["parameters"]}
        pb = {p["parameterId"]: p for p in vb["parameters"]}
        param_diff = {
            "added": sorted(set(pb) - set(pa)),
            "removed": sorted(set(pa) - set(pb)),
            "changed": sorted(pid for pid in set(pa) & set(pb) if pa[pid] != pb[pid]),
        }
        # Fit comparison from each version's stored metrics (present for fitted versions).
        na, nb = va.get("neg_log_likelihood"), vb.get("neg_log_likelihood")
        fit = {
            "from_neg_log_likelihood": na,
            "to_neg_log_likelihood": nb,
            # positive => version `b` fits better than `a` (lower neg-log-likelihood)
            "neg_log_likelihood_delta": (na - nb) if (na is not None and nb is not None) else None,
            "from_chi2": va.get("chi2"),
            "to_chi2": vb.get("chi2"),
        }
        return {
            "from": a,
            "to": b,
            "model_diff": model_diff(
                va["model_antimony"], vb["model_antimony"], f"v{a}/model.ant", f"v{b}/model.ant"
            ),
            "parameter_diff": param_diff,
            "fit": fit,
        }
