"""Per-store log routing (``logging_setup.use_store``).

One process hosts many sessions, and each session's ``petab_mcp.log`` and ``pypesto.log`` must hold
that session's records only: ``pypesto.log`` is what ``get_fitting_log`` serves the agent, so a
record in the wrong store is a wrong answer about the fit. Records go through the real loggers and
are read straight off disk without a flush, which is what ``get_fitting_log`` and
``test_engine._log_text`` rely on. Every scope here is a ``with`` block, so none leaks into the
next test.
"""

from __future__ import annotations

import contextvars
import logging
import threading
from concurrent.futures import ThreadPoolExecutor

from petab_mcp import logging_setup as L


def _store(root):
    """Configure both logger groups for ``root``, as ``PetabSession.__init__`` does."""
    L.configure(root)
    L.configure_pypesto(root)
    return root


def _log(tag: str) -> None:
    logging.getLogger("petab_mcp").info("record from %s", tag)
    logging.getLogger("pypesto").debug("record from %s", tag)


def _text(root, name: str = "petab_mcp.log") -> str:
    return (root / name).read_text(encoding="utf-8")


def test_two_sessions_logging_at_once_each_write_only_their_own_store(tmp_path):
    """The regression: ONE file handler per logger group, swapped to the most recently configured
    store, so a second session redirected every session's logs, and the first session's
    ``get_fitting_log`` then served the second's optimizer output. Two threads log interleaved
    through the same two handlers, each inside its own ``use_store``."""
    a, b = _store(tmp_path / "a"), _store(tmp_path / "b")
    both_inside = threading.Barrier(2, timeout=10)

    def session(root, tag):
        with L.use_store(root):
            both_inside.wait()
            for _ in range(50):
                _log(tag)

    with ThreadPoolExecutor(2) as pool:
        for call in [pool.submit(session, a, "A"), pool.submit(session, b, "B")]:
            call.result()
    for root, own, other in [(a, "A", "B"), (b, "B", "A")]:
        for name in ("petab_mcp.log", "pypesto.log"):
            text = _text(root, name)
            assert text.count(f"record from {own}") == 50
            assert f"record from {other}" not in text


def test_without_use_store_records_go_to_the_last_configured_store(tmp_path):
    """Single-session use never calls ``use_store``, and every test that drives a ``PetabSession``
    directly reads its log back from the store it configured last."""
    a, b = _store(tmp_path / "a"), _store(tmp_path / "b")
    _log("outside")
    assert "record from outside" in _text(b) + _text(b, "pypesto.log")
    assert "record from outside" not in _text(a) + _text(a, "pypesto.log")


def test_a_plain_use_store_ends_with_its_context(tmp_path):
    """A plain ``use_store`` (no ``with``) is meant for the top of a sync tool call, and is safe
    there only because FastMCP runs each such call in its own context copy: the route must not
    outlive the call into whatever the thread runs next."""
    a, b = _store(tmp_path / "a"), _store(tmp_path / "b")

    def call():
        L.use_store(a)
        _log("inside")

    contextvars.copy_context().run(call)
    _log("after")
    assert "record from inside" in _text(a)
    assert "record from after" in _text(b)
    assert "record from after" not in _text(a)


def test_fides_records_land_in_pypesto_log(tmp_path):
    """Fides logs through its own ``fides`` logger; its records belong in the fit log beside
    pyPESTO's, not in ``petab_mcp.log`` and not, by propagation, on the server's stderr."""
    a = _store(tmp_path / "a")
    with L.use_store(a):
        logging.getLogger("fides").warning("trust region collapsed")
    assert "trust region collapsed" in _text(a, "pypesto.log")
    assert "trust region collapsed" not in _text(a)
    assert logging.getLogger("fides").propagate is False


def test_both_files_exist_right_after_configure(tmp_path):
    """``get_fitting_log`` reports ``exists`` before any fit has logged a line, so ``configure*``
    create the files rather than leaving them to the first record."""
    a = _store(tmp_path / "a")
    assert (a / "petab_mcp.log").is_file()
    assert L.read_pypesto_log(a, 0)["exists"] is True


def test_a_store_closed_by_the_lru_is_reopened_for_appending(tmp_path, monkeypatch):
    """Open files are capped, since pytest builds hundreds of sessions per process. A store whose
    file was closed to make room must get its next record appended, not dropped or truncated."""
    monkeypatch.setattr(L, "_MAX_OPEN", 1)
    a, b = _store(tmp_path / "a"), _store(tmp_path / "b")
    for root, tag in [(a, "a1"), (b, "b1"), (a, "a2")]:
        with L.use_store(root):
            _log(tag)
    for name in ("petab_mcp.log", "pypesto.log"):
        assert "record from a1" in _text(a, name) and "record from a2" in _text(a, name)
        assert "record from b1" in _text(b, name)
    assert len(L._PETAB_LOG._files) == len(L._PYPESTO_LOG._files) == 1
