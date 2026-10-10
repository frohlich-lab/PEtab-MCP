"""The rate limit on AMICI's own warnings (``logging_setup.configure_amici``).

AMICI logs one warning per non-finite value and the solver re-evaluates a failing right-hand
side many times per step, so one bad start emits the same line without bound: the 2026-09-04
sweep left a >5 GB ``server.log`` that was ~28M copies of two ``AMICI:NaN`` lines. What these
lock in is the throttle itself, and the three things it quietly depends on — that we own the
`amici` logger instead of AMICI's unfiltered one, that the filter sits on the HANDLER so it sees
records propagating from `amici.sim.sundials._swig_wrappers`, and that a forked pyPESTO worker
inherits it.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from petab_mcp import logging_setup as L
from petab_mcp.session import PetabSession

_NAN_XDOT = (
    "[condition_step_30_0][AMICI:NaN] AMICI encountered a NaN value for xdot[2] (A) at t=%f "
)


def _emit(
    filt: L._RateLimitFilter, template: str, n: int, second: int = 1_700_000_000
) -> list[str]:
    """Push ``n`` occurrences, all timestamped within wall-clock ``second``, through the
    filter; return the messages that survived."""
    kept = []
    for i in range(n):
        record = logging.LogRecord(
            "amici.sim.sundials._swig_wrappers",
            logging.WARNING,
            __file__,
            1,
            template,
            (3599.999511 + i * 1e-6,),
            None,
        )
        record.created = second + i * 1e-6  # same second, sub-second jitter like real retries
        if filt.filter(record):
            kept.append(record.getMessage())
    return kept


def test_a_repeated_warning_within_one_second_is_throttled():
    """28M lines logged inside one second must cost a handful."""
    kept = _emit(L._RateLimitFilter(), _NAN_XDOT, 30_000)
    assert len(kept) == L._AMICI_PER_SECOND
    assert kept[0].endswith("at t=3599.999511 ")  # untouched, still greppable as itself


def test_a_new_second_gets_a_fresh_budget():
    """The budget resets per wall-clock second, so a warning that persists across seconds keeps
    being reported rather than going silent forever after the first second's cap."""
    filt = L._RateLimitFilter()
    _emit(filt, _NAN_XDOT, 100, second=1_700_000_000)
    kept = _emit(filt, _NAN_XDOT, 100, second=1_700_000_001)
    assert len(kept) == L._AMICI_PER_SECOND


def _run(body: str) -> str:
    """Run ``body`` in a fresh interpreter — import order is the thing under test."""
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(body)],
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "PYTHONWARNINGS": "ignore"},
    )
    assert out.returncode == 0, out.stderr
    return out.stdout


def test_configure_amici_pre_empts_amicis_own_handler():
    """Load-bearing and invisible: `amici.logging.get_logger` sets up its own unfiltered
    StreamHandler only if the `amici` logger does not exist yet. Naming it first is what makes
    ours the only one — so this breaks silently if the call ever moves after an AMICI import.
    """
    assert (
        _run("""
        import logging, sys
        from petab_mcp import logging_setup as L
        assert "amici" not in sys.modules, "amici imported before configure_amici"
        L.configure_amici()
        import amici.logging
        lg = logging.getLogger("amici")
        print(lg.handlers == [L._amici_handler], lg.propagate)
    """).strip()
        == "True False"
    )


def test_a_forked_worker_inherits_the_filter():
    """pyPESTO's MultiProcessEngine forks, and the fits that provoke these warnings run in those
    workers. The parent-side handler only helps because it is inherited — the reason this works
    where the equivalent fix for `fides` could not (see `petab_mcp.fitting`)."""
    out = _run("""
        import logging, multiprocessing as mp, sys
        from petab_mcp import logging_setup as L
        L.configure_amici()
        logging.getLogger("amici").handlers[0].setStream(sys.stdout)

        def work(_):
            log = logging.getLogger("amici.sim.sundials._swig_wrappers")
            for i in range(2000):
                record = log.makeRecord(
                    log.name, logging.WARNING, "<sim>", 1,
                    "[AMICI:NaN] NaN value for xdot[2] (A) at t=%f", (1.0 + i * 1e-6,), None,
                )
                record.created = 1_700_000_000  # one wall-clock second, as a retry storm is
                log.handle(record)

        if __name__ == "__main__":
            with mp.get_context("fork").Pool(2) as pool:
                pool.map(work, range(2))
    """)
    # Each worker is capped at _AMICI_PER_SECOND rather than emitting 2000. The second is fixed
    # because real time let a burst straddle a second boundary on a loaded CI runner, doubling
    # the cap.
    assert len(out.splitlines()) <= 2 * L._AMICI_PER_SECOND


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ERROR", logging.ERROR),
        ("error", logging.ERROR),
        ("10", logging.DEBUG),
        ("", logging.WARNING),
        ("VERBOSE", logging.WARNING),
    ],
)
def test_amici_log_env_var_still_works(monkeypatch, raw, expected):
    """AMICI's own knob must survive us taking over its logger — including its NAMED form, which
    is what AMICI documents and what a person actually types. Parsing it as a bare `int` took the
    server down in `PetabSession.__init__` before a valid `AMICI_LOG=ERROR` could reach a fit."""
    monkeypatch.setenv("AMICI_LOG", raw)
    assert L._amici_level() == expected


def test_a_bad_amici_log_value_does_not_take_the_server_down(monkeypatch, tmp_path):
    """Deliberately more lenient than AMICI, which raises on an unparseable value. This runs once
    per session construction on a batch node, so a raise is not a bad log level — it is every
    task in the sweep failing to start. Drive the real constructor, since that is where it broke.
    """
    monkeypatch.setenv("AMICI_LOG", "not-a-level")
    shutil.copytree(Path(__file__).parent / "data" / "toy_v2", tmp_path / "p")
    PetabSession(str(tmp_path / "p" / "problem.yaml"))
    assert logging.getLogger("amici").level == logging.WARNING
