from __future__ import annotations

import pytest

from petab_mcp.references import (
    EutilsReferenceResolver,
    ReferenceError,
    StubReferenceResolver,
    resolve_references,
)


def test_stub_resolves_pmid_and_doi():
    r = StubReferenceResolver().resolve("PMID:12345")
    assert r.pmid == "12345"
    assert r.verified is True
    d = StubReferenceResolver().resolve("https://doi.org/10.1038/nature14539")
    assert d.doi == "10.1038/nature14539"


def test_resolve_references_requires_nonempty():
    resolver = StubReferenceResolver()
    with pytest.raises(ReferenceError):
        resolve_references(resolver, [])
    with pytest.raises(ReferenceError):
        resolve_references(resolver, ["   "])
    recs = resolve_references(resolver, ["PMID:1", "10.1/x"])
    assert len(recs) == 2
    assert recs[0]["input"] == "PMID:1"


def test_eutils_off_mode_no_network():
    """'off' mode records references unverified without any network call."""
    resolver = EutilsReferenceResolver(mode="off")
    r = resolver.resolve("PMID:99999")
    assert r.verified is False
    assert r.pmid == "99999"
    assert "off" in (r.note or "")


def test_eutils_bad_mode():
    with pytest.raises(ValueError):
        EutilsReferenceResolver(mode="bogus")
