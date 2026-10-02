"""The process-wide libantimony lock (antimony_guard).

The 2026-08-15 sneyd A/B lost 20/20 draft-arm runs to signal 11: three FastMCP worker threads
simultaneously inside ``antimony.clearPreviousLoads`` (libantimony state is process-global and
not thread-safe). A segfault cannot be asserted from inside the process, so these tests pin the
structural properties that prevent it: the lock is installed on both conversion entry points,
idempotently, and stays live under concurrency.
"""

from __future__ import annotations

from petab.v2.models.sbml_model import SbmlModel

import petab_mcp  # noqa: F401  - importing the package installs the guard (__init__)
from petab_mcp import antimony_guard

_ANT = (
    "model t{i}\n  compartment c = 1;\n  species A in c;\n  A = 1; k = 0.1;\n"
    "  r: A -> ; k*A;\nend\n"
)


def test_guard_is_installed_by_package_import():
    assert getattr(SbmlModel, antimony_guard._INSTALLED_FLAG, False)
    # idempotent: a second install must not re-wrap — a wrapper-of-wrapper chain growing per
    # import would be a leak, and with a non-reentrant Lock it would deadlock on the first call
    before_to, before_from = SbmlModel.to_antimony, SbmlModel.from_antimony
    antimony_guard.install()
    assert SbmlModel.to_antimony is before_to
    assert SbmlModel.from_antimony is before_from


def test_wrapped_conversions_still_round_trip():
    m = SbmlModel.from_antimony(_ANT.format(i=0))
    ant = m.to_antimony()
    assert "r:" in ant and "k" in ant


def test_conversions_do_not_deadlock_under_concurrency():
    """The guard is a NON-reentrant Lock, so a nested conversion would hang a worker thread
    forever rather than corrupt state (see the module docstring: verified no nesting exists in
    the installed petab, but a future petab/amici path could introduce one).

    Small and timeout-based on purpose: it asserts LIVENESS, which is the failure mode the Lock
    choice trades for, and it fails as a clean assert rather than by crashing the xdist worker.
    """
    import threading

    done = []

    def work(i: int) -> None:
        for _ in range(3):
            SbmlModel.from_antimony(_ANT.format(i=i)).to_antimony()
        done.append(i)

    threads = [threading.Thread(target=work, args=(i,), daemon=True) for i in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert sorted(done) == [0, 1, 2], (
        f"conversions did not all complete ({sorted(done)}) — a nested conversion under the "
        "non-reentrant ANTIMONY_LOCK deadlocks; see antimony_guard's module docstring"
    )
