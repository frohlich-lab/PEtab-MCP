"""Reference resolution and validation against PubMed.

Every model operation requires a non-empty justification; references are OPTIONAL. Any
references that ARE supplied are validated against PubMed so provenance points at real
literature. Because this package is a standalone MCP server (it cannot call another MCP
server), it talks to NCBI E-utilities directly over HTTPS via ``urllib`` — no extra dependency.

Validation is pluggable (``ReferenceResolver``) and mode-controlled:

* ``required``    — a reference that cannot be resolved rejects the operation.
* ``best_effort`` — (default) verify when the network allows; otherwise record the
  reference as unverified and proceed. Keeps the server usable behind restrictive
  proxies. References must still be *present*.
* ``off``         — do not contact the network; references are recorded unverified.

Tests use :class:`StubReferenceResolver`.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any

from .constants import REFERENCE_TIMEOUT_SECONDS
from .errors import PetabMcpError

_EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"

_PMID_RE = re.compile(r"^\s*(?:pmid[:\s]*)?(\d{1,9})\s*$", re.IGNORECASE)
_DOI_RE = re.compile(r"(10\.\d{4,9}/[^\s\"'<>]+)", re.IGNORECASE)


class ReferenceError(PetabMcpError):
    """A required reference is missing or could not be validated."""


@dataclass
class ResolvedReference:
    input: str
    verified: bool
    pmid: str | None = None
    doi: str | None = None
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: str | None = None
    journal: str | None = None
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StubReferenceResolver:
    """Deterministic resolver for tests (no network)."""

    def __init__(self, verified: bool = True) -> None:
        self.verified = verified

    def resolve(self, ref: str) -> ResolvedReference:
        pmid = None
        m = _PMID_RE.match(ref)
        if m:
            pmid = m.group(1)
        doi_m = _DOI_RE.search(ref)
        return ResolvedReference(
            input=ref,
            verified=self.verified,
            pmid=pmid,
            doi=doi_m.group(1) if doi_m else None,
            title="Stubbed reference" if self.verified else None,
            note=None if self.verified else "stub: unverified",
        )


class EutilsReferenceResolver:
    """Resolve PMIDs/DOIs against NCBI E-utilities."""

    def __init__(self, mode: str = "best_effort", timeout: float = REFERENCE_TIMEOUT_SECONDS) -> None:
        if mode not in ("required", "best_effort", "off"):
            raise ValueError(f"Unknown reference-validation mode {mode!r}.")
        self.mode = mode
        self.timeout = timeout

    # -- network helpers ---------------------------------------------------
    def _get_json(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(url, headers={"User-Agent": "petab-mcp/0.1"})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def _doi_to_pmid(self, doi: str) -> str | None:
        url = _IDCONV + "?" + urllib.parse.urlencode(
            {"ids": doi, "format": "json", "tool": "petab-mcp", "email": "petab-mcp@example.org"}
        )
        data = self._get_json(url)
        for rec in data.get("records", []):
            if rec.get("pmid"):
                return str(rec["pmid"])
        # fallback: esearch by DOI
        url = _EUTILS + "/esearch.fcgi?" + urllib.parse.urlencode(
            {"db": "pubmed", "term": f"{doi}[AID]", "retmode": "json"}
        )
        data = self._get_json(url)
        ids = data.get("esearchresult", {}).get("idlist", [])
        return str(ids[0]) if ids else None

    def _summary(self, pmid: str) -> dict[str, Any] | None:
        url = _EUTILS + "/esummary.fcgi?" + urllib.parse.urlencode(
            {"db": "pubmed", "id": pmid, "retmode": "json"}
        )
        data = self._get_json(url)
        result = data.get("result", {})
        return result.get(pmid)

    # -- public API --------------------------------------------------------
    def resolve(self, ref: str) -> ResolvedReference:
        pmid_m = _PMID_RE.match(ref)
        doi_m = _DOI_RE.search(ref)
        doi = doi_m.group(1) if doi_m else None

        if self.mode == "off":
            return ResolvedReference(
                input=ref,
                verified=False,
                pmid=pmid_m.group(1) if pmid_m else None,
                doi=doi,
                note="validation off",
            )

        # Resolve the PMID up front so a later network failure in ``_summary`` still keeps a
        # DOI-resolved PMID (the except branch below reports ``pmid``, not just ``pmid_m``).
        pmid = pmid_m.group(1) if pmid_m else None
        try:
            if pmid is None and doi:
                pmid = self._doi_to_pmid(doi)
            summary = self._summary(pmid) if pmid else None
            if summary:
                return ResolvedReference(
                    input=ref,
                    verified=True,
                    pmid=pmid,
                    doi=doi,
                    title=summary.get("title"),
                    authors=[a.get("name") for a in summary.get("authors", []) if a.get("name")],
                    year=(summary.get("pubdate") or "")[:4] or None,
                    journal=summary.get("fulljournalname") or summary.get("source"),
                )
            # reachable but not found
            note = "no PubMed record found"
            if self.mode == "required":
                raise ReferenceError(
                    f"Reference {ref!r} could not be resolved to a PubMed record."
                )
            return ResolvedReference(input=ref, verified=False, pmid=pmid, doi=doi, note=note)
        except ReferenceError:
            raise
        except Exception as exc:  # network error
            if self.mode == "required":
                raise ReferenceError(
                    f"Cannot verify reference {ref!r} (network error: {exc}). "
                    "Set reference validation to 'best_effort' or 'off' to proceed."
                ) from exc
            return ResolvedReference(
                input=ref,
                verified=False,
                pmid=pmid,
                doi=doi,
                note=f"unverified (network unavailable: {type(exc).__name__})",
            )


def resolve_references(
    resolver, references: list[str] | None, required: bool = True
) -> list[dict[str, Any]]:
    """Resolve references, optionally requiring at least one.

    Returns the resolved records (dicts). When ``required`` is True, an absent/empty
    list raises ``ReferenceError``; when False, an empty list resolves to ``[]``. In the
    resolver's ``required`` mode an unresolvable reference always raises.
    """
    references = references or []
    if not isinstance(references, list):
        raise ReferenceError("`references` must be a list of citation strings.")
    cleaned = [r.strip() for r in references if isinstance(r, str) and r.strip()]
    if not cleaned:
        if required:
            raise ReferenceError("At least one reference is required for this operation.")
        return []
    return [resolver.resolve(r).to_dict() for r in cleaned]
