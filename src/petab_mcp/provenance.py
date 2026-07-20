"""Provenance and versioning, persisted to disk.

Under ``<store_root>/`` (default ``<problem_dir>/.petab_mcp``):

* ``provenance.jsonl`` — append-only event log (alterations, batch commits, rollbacks,
  resets). Each event carries a UTC timestamp.
* ``versions/vN/`` — a snapshot of the mutable model state at version N: ``model.ant``
  (the canonical Antimony) and ``parameters.json`` (the parameter-table records), plus
  ``meta.json`` (label, kind, parent, timestamp). A fitted version also carries
  ``simulations.json`` (the measured-vs-simulated dataframe + residuals), ``optimization.json``
  (the pyPESTO per-start + per-iteration trace records), and ``pypesto_result.hdf5`` (the
  pyPESTO result file, copied in from the per-model cache) — so rolling back restores a
  version's fit artifacts, not just its model.
* ``cache/`` — pyPESTO result files keyed by SBML hash (written on each fit, copied into
  version dirs when snapshotted).
* ``HEAD`` — the current version number.

The store is pure storage: it records Antimony text + parameter records and reads them
back. Applying a version to the live problem lives in the session/operations layer.
"""

from __future__ import annotations

import difflib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_simulations_tsv(vdir: Path, records: list[dict[str, Any]]) -> None:
    """Best-effort PEtab simulation table for interop (never read back)."""
    if not records:
        return
    try:
        import pandas as pd
        from petab.v1.C import (
            OBSERVABLE_ID,
            SIMULATION,
            SIMULATION_CONDITION_ID,
            TIME,
        )

        rows = [
            {
                OBSERVABLE_ID: r.get("observable_id"),
                SIMULATION_CONDITION_ID: r.get("condition_id"),
                TIME: r.get("time"),
                SIMULATION: r.get("simulation"),
            }
            for r in records
        ]
        pd.DataFrame(rows).to_csv(vdir / "simulations.tsv", sep="\t", index=False)
    except Exception:  # interop convenience only — never fail a snapshot over it
        pass


class ProvenanceStore:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.versions_dir = self.root / "versions"
        self.log_path = self.root / "provenance.jsonl"
        self.errors_path = self.root / "errors.jsonl"
        self.head_path = self.root / "HEAD"
        self.versions_dir.mkdir(parents=True, exist_ok=True)

    # -- event log ---------------------------------------------------------
    def append_event(self, event: dict[str, Any]) -> dict[str, Any]:
        event = {"timestamp": _now(), **event}
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event) + "\n")
        return event

    def recent_events(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._tail(self.log_path, limit)

    # -- error log (separate from the success-oriented provenance log) ------
    def log_error(self, record: dict[str, Any]) -> dict[str, Any]:
        record = {"timestamp": _now(), **record}
        with self.errors_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
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

    # -- versions ----------------------------------------------------------
    def snapshot_version(
        self,
        model_ant: str,
        parameters: Any,
        label: str | None = None,
        kind: str = "snapshot",
        parent: int | None = None,
        meta: dict[str, Any] | None = None,
        simulation: dict[str, Any] | None = None,
        optimization: dict[str, Any] | None = None,
    ) -> int:
        n = self.current_version + 1
        vdir = self.versions_dir / f"v{n}"
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "model.ant").write_text(model_ant, encoding="utf-8")
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
        (vdir / "meta.json").write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")
        # The simulation dataframe generated for this version (measured vs simulated +
        # residuals). Stored as JSON (read back by get_version) plus a best-effort PEtab
        # simulation table for interop (never read back).
        if simulation is not None:
            (vdir / "simulations.json").write_text(json.dumps(simulation, indent=2), encoding="utf-8")
            _write_simulations_tsv(vdir, simulation.get("records") or [])
        # The optimization traces/results for this version: the per-start + per-iteration
        # records (read back into the DuckDB tables) as JSON, plus the pyPESTO HDF5 result
        # file copied in from the per-model cache (or the source version, on rollback).
        if optimization is not None:
            self._write_optimization(vdir, optimization)
        self._set_head(n)
        return n

    @staticmethod
    def _write_optimization(vdir: Path, optimization: dict[str, Any]) -> None:
        persist = {k: optimization.get(k) for k in ("n_starts", "starts", "trace")}
        (vdir / "optimization.json").write_text(json.dumps(persist, indent=2), encoding="utf-8")
        result_file = optimization.get("result_file")
        if result_file:
            try:
                src = Path(result_file)
                if src.is_file():
                    shutil.copyfile(src, vdir / "pypesto_result.hdf5")
            except OSError:  # copying the interop hdf5 must never fail a snapshot
                pass

    def _version_dir(self, n: int) -> Path:
        vdir = self.versions_dir / f"v{n}"
        if not vdir.is_dir():
            from .errors import UnknownEntityError

            raise UnknownEntityError(f"No such version: v{n}")
        return vdir

    def get_parameter_df(self, n: int):
        """Read a version's parameter table back as a DataFrame (index = ``parameterId``)."""
        return self._read_parameter_df(self._version_dir(n))

    @staticmethod
    def _read_parameter_df(vdir: Path):
        """Load ``parameters.tsv`` as a DataFrame, falling back to legacy ``parameters.json``."""
        import pandas as pd

        tsv = vdir / "parameters.tsv"
        if tsv.is_file():
            return pd.read_csv(tsv, sep="\t", index_col=0)
        legacy = vdir / "parameters.json"  # versions written before the TSV switch
        df = pd.DataFrame(json.loads(legacy.read_text()))
        if "parameterId" in df.columns:
            df = df.set_index("parameterId")
        return df

    def get_version(self, n: int) -> dict[str, Any]:
        vdir = self._version_dir(n)
        meta = json.loads((vdir / "meta.json").read_text())
        sim_path = vdir / "simulations.json"
        simulation = json.loads(sim_path.read_text()) if sim_path.is_file() else None
        opt_path = vdir / "optimization.json"
        optimization = json.loads(opt_path.read_text()) if opt_path.is_file() else None
        if optimization is not None:
            # Point at this version's copy of the pyPESTO result file (if it was stored), so a
            # later rollback can copy it forward and tools can locate it.
            hdf5 = vdir / "pypesto_result.hdf5"
            optimization["result_file"] = str(hdf5) if hdf5.is_file() else None
        # `parameters` stays a list of JSON records — the shape the MCP tool and diff want
        # (and it must be JSON-serializable). The raw DataFrame is available separately via
        # `get_parameter_df`, which the session uses to restore state directly.
        from .parameters import parameter_df_to_records

        return {
            **meta,
            "model_antimony": (vdir / "model.ant").read_text(encoding="utf-8"),
            "parameters": parameter_df_to_records(self._read_parameter_df(vdir)),
            "simulation": simulation,
            "optimization": optimization,
        }

    def list_versions(self) -> list[dict[str, Any]]:
        out = []
        # Only ``v<int>`` directories are versions; skip any stray ``v*`` entry (a leftover temp
        # dir, an editor artefact) so `int(name[1:])` can't raise and break version listing.
        vdirs = [d for d in self.versions_dir.glob("v*") if d.is_dir() and d.name[1:].isdigit()]
        for vdir in sorted(vdirs, key=lambda p: int(p.name[1:])):
            meta = json.loads((vdir / "meta.json").read_text())
            out.append(meta)
        return out

    def diff_versions(self, a: int, b: int) -> dict[str, Any]:
        va, vb = self.get_version(a), self.get_version(b)
        model_diff = "".join(
            difflib.unified_diff(
                va["model_antimony"].splitlines(keepends=True),
                vb["model_antimony"].splitlines(keepends=True),
                fromfile=f"v{a}/model.ant",
                tofile=f"v{b}/model.ant",
            )
        )
        pa = {p["parameterId"]: p for p in va["parameters"]}
        pb = {p["parameterId"]: p for p in vb["parameters"]}
        param_diff = {
            "added": sorted(set(pb) - set(pa)),
            "removed": sorted(set(pa) - set(pb)),
            "changed": sorted(
                pid for pid in set(pa) & set(pb) if pa[pid] != pb[pid]
            ),
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
        return {"from": a, "to": b, "model_diff": model_diff,
                "parameter_diff": param_diff, "fit": fit}
