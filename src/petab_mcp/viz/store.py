"""The version store as the viewer reads it: every file opened read-only, nothing written.

A store is the directory :mod:`petab_mcp.provenance` documents. The viewer also reads one a harness
archived (gemot's ``_runs_archive/<run>/.petab_mcp``), which keeps fewer files: no ``FORMAT``, no
``pypesto_result.hdf5`` unless asked, and ``optimization.json`` gzipped as ``optimization.json.gz``.
So each read is of what is there, and a missing file is ``None``, never an error. A store whose
``FORMAT`` names another layout is refused, as the server refuses it.
"""

from __future__ import annotations

import gzip
import json
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd

from ..errors import ValidationError
from ..provenance import FIT_PROGRESS, STORE_FORMAT

#: The version directories, ``versions/v<N>``.
_VERSION_DIR = re.compile(r"v(\d+)")

#: The fields of a version's ``meta.json`` the version table carries.
_META_FIELDS = (
    "kind",
    "label",
    "parent",
    "timestamp",
    "fit_status",
    "criterion",
    "criterion_value",
    "neg_log_likelihood",
    "chi2",
    "k",
    "n_starts",
    "n_starts_killed",
    "is_new_best",
    "wall_seconds",
)


def _read_json(path: Path) -> Any:
    """``path`` as JSON, or its gzipped copy ``<path>.gz``; ``None`` when neither is there."""
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    gz = path.with_name(path.name + ".gz")
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _finite(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


class Store:
    """One store, read on demand. Version files never change once written, so each is read once
    and kept, but only once it is there: the server writes a version's ``meta.json`` before its
    simulation, and a file read missing mid-write is read again next time. ``HEAD``, the event log
    and the fit's progress are read again on every call."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        if not (self.root / "versions").is_dir():
            raise ValidationError(f"{self.root} is not a version store: it has no versions/.")
        format_path = self.root / "FORMAT"
        if format_path.exists() and (found := format_path.read_text().strip()) != STORE_FORMAT:
            raise ValidationError(
                f"{self.root} is a {found!r} store, not {STORE_FORMAT!r}: an earlier server wrote "
                "it and its numbers may not mean what they mean now."
            )
        self._cache: dict[tuple[str, int], Any] = {}

    # -- what changes --------------------------------------------------------------------------
    def signature(self) -> tuple:
        """What a live viewer polls: changes whenever a version, an event or the fit's progress
        does."""

        def stamp(path: Path) -> float | None:
            try:
                return path.stat().st_mtime_ns
            except OSError:
                return None

        return (
            stamp(self.root / "HEAD"),
            stamp(self.root / "provenance.jsonl"),
            stamp(self.root / FIT_PROGRESS),
            tuple(self.versions()),
        )

    def head(self) -> int | None:
        try:
            return int((self.root / "HEAD").read_text().strip())
        except (OSError, ValueError):
            versions = self.versions()
            return versions[-1] if versions else None

    def versions(self) -> list[int]:
        return sorted(
            int(m.group(1))
            for p in (self.root / "versions").iterdir()
            if p.is_dir() and (m := _VERSION_DIR.fullmatch(p.name)) and (p / "meta.json").exists()
        )

    def events(self) -> list[dict[str, Any]]:
        path = self.root / "provenance.jsonl"
        if not path.exists():
            return []
        out = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:  # a line the server is still writing
                continue
        return out

    def fit_progress(self) -> dict[str, Any] | None:
        """The running change's fit (:class:`petab_mcp.multistart.Progress`), ``None`` when no
        change runs."""
        try:
            return _read_json(self.root / FIT_PROGRESS)
        except (OSError, json.JSONDecodeError):
            return None

    # -- one version ---------------------------------------------------------------------------
    def _version_file(self, version: int, name: str, reader) -> Any:
        key = (name, version)
        if key in self._cache:
            return self._cache[key]
        value = reader(self.root / "versions" / f"v{version}" / name)
        if value is not None and not (isinstance(value, dict) and not value):
            self._cache[key] = value
        return value

    def meta(self, version: int) -> dict[str, Any]:
        return self._version_file(version, "meta.json", _read_json) or {}

    def parameters(self, version: int) -> pd.DataFrame | None:
        def read(path: Path) -> pd.DataFrame | None:
            if not path.exists():
                return None
            return pd.read_csv(path, sep="\t", dtype={"parameterId": str}).set_index("parameterId")

        return self._version_file(version, "parameters.tsv", read)

    def simulations(self, version: int) -> pd.DataFrame | None:
        """The version's per-point records: measurement, simulation and residuals."""
        sim = self._version_file(version, "simulations.json", _read_json)
        if not sim or not sim.get("records"):
            return None
        df = pd.DataFrame(sim["records"])
        df["experiment_id"] = df["experiment_id"].fillna("").astype(str)
        return df

    def optimization(self, version: int) -> dict[str, Any] | None:
        return self._version_file(version, "optimization.json", _read_json)

    def starts(self, version: int) -> pd.DataFrame | None:
        """One row per start, best first: its objective, effort and, where the version kept its
        pyPESTO result file, how it ended (``exitflag``, ``message``) and where it began (``x0``).
        ``x`` and ``x0`` are on the estimation scale, in ``optimization.json``'s ``x_names``
        order."""
        opt = self.optimization(version)
        if not opt or not opt.get("starts"):
            return None
        df = pd.DataFrame(opt["starts"])
        df["id"] = df["id"].astype(str)
        df["fval"] = [_finite(v) for v in df["fval"]]
        ended = self._ended(version)
        for column in ("exitflag", "message", "x0"):
            df[column] = [ended.get(i, {}).get(column) for i in df["id"]]
        return df.sort_values("fval", na_position="last").reset_index(drop=True)

    def _ended(self, version: int) -> dict[str, dict[str, Any]]:
        """Each start's ``exitflag``, ``message`` and ``x0`` by id, from ``pypesto_result.hdf5``;
        empty where the version has none. Read with h5py, in the layout pyPESTO's storage writes
        (``optimization/results/<n>``, its fields as attributes, ``x0`` a dataset), rather than
        through pyPESTO, whose import loads AMICI: the viewer never simulates, and loads nothing
        that could."""

        def read(path: Path) -> dict[str, dict[str, Any]]:
            if not path.exists():
                return {}
            import h5py

            out = {}
            try:
                with h5py.File(path, "r") as fh:
                    for group in fh["optimization/results"].values():
                        attrs = group.attrs
                        flag = attrs.get("exitflag")
                        message = attrs.get("message")
                        out[str(attrs.get("id"))] = {
                            "exitflag": None if flag is None else int(flag),
                            "message": message.decode() if isinstance(message, bytes) else message,
                            "x0": [float(v) for v in group["x0"][()]] if "x0" in group else None,
                        }
            except (OSError, KeyError):  # an archive's or another pyPESTO's file
                return {}
            return out

        return self._version_file(version, "pypesto_result.hdf5", read)

    def trace(self, version: int) -> pd.DataFrame | None:
        opt = self.optimization(version)
        if not opt or not opt.get("trace"):
            return None
        return pd.DataFrame(opt["trace"])

    # -- across versions -----------------------------------------------------------------------
    def table(self) -> pd.DataFrame:
        """One row per version: its kind, parent, score and the justification of the change that
        made it (from the event log)."""
        why = {
            e["version"]: e.get("justification")
            for e in self.events()
            if e.get("type") == "fit" and "version" in e
        }
        rows = []
        for v in self.versions():
            meta = self.meta(v)
            rows.append(
                {
                    "version": v,
                    **{k: meta.get(k) for k in _META_FIELDS},
                    "justification": why.get(v),
                }
            )
        df = pd.DataFrame(rows, columns=["version", *_META_FIELDS, "justification"])
        df["criterion_value"] = [_finite(v) for v in df["criterion_value"]]
        return df

    def best(self) -> int | None:
        """The best version as the server judged it: the last one its fit stamped ``is_new_best``
        (the checkpoint can keep a version over a lower score, see
        :meth:`petab_mcp.session.PetabSession.record_fit_progress`)."""
        df = self.table()
        flagged = df[df["is_new_best"].fillna(False).astype(bool)]
        return None if flagged.empty else int(flagged["version"].iloc[-1])
