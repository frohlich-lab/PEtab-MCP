"""Serialize every libantimony conversion in this process behind one lock.

libantimony keeps PROCESS-GLOBAL state: every ``loadSBMLString``/``getAntimonyString`` round
trip mutates the same module-level document table, and ``clearPreviousLoads`` walks it. It is
not thread-safe, and FastMCP dispatches each tool call on its own worker thread — so two
concurrent tool calls that both convert (in the draft era: an ``edit_draft`` linting itself
while a ``get_draft`` rendered its review diff) race inside the C library and segfault the
whole server.

Measured, not hypothetical: the 2026-08-15 sneyd A/B lost 20 of 20 draft-arm runs to signal 11,
and the faulthandler dump (enabled in ``runner/main.py`` for exactly this bug) shows THREE
threads simultaneously in ``antimony.py:211 clearPreviousLoads`` under
``SbmlModel.to_antimony``. The same crash had been killing occasional draft-arm runs since the
first sweep (the "dsf3 segfaults", 5 runs, ~35M tokens) — the draft arm just multiplied the
conversion rate until the race was near-certain.

The lock is installed by PATCHING ``SbmlModel.from_antimony`` / ``to_antimony`` once, rather
than by wrapping call sites: there are ~18 call sites across session/operations/engine and
any future one would silently reopen the race. It wraps the two CONVERSION SEQUENCES rather than
the ``antimony`` module's individual functions on purpose — one conversion is
``clearPreviousLoads`` -> ``loadSBMLString`` -> ``getAntimonyString``, and locking each call
separately would still let two sequences interleave, which is the actual corruption.

WHAT IS NOT COVERED, so nobody reads more into this than it does: code that drives ``antimony``
directly instead of through ``SbmlModel`` (``petab_harness/prune_fixed_terms.py`` does its own
``import antimony`` round trip), and petab's/amici's module-level ``antimony2sbml`` helpers. Any
such caller must take :data:`ANTIMONY_LOCK` itself — it is exported for that. The lock also only
exists once ``petab_mcp`` has been imported, so a pure-harness process gets no protection.

A plain Lock, not an RLock. VERIFIED no nested conversion in the installed petab: from_antimony
is ``antimony2sbml`` + ``SbmlModel.from_string`` (libsbml only) and to_antimony is
``to_sbml_str`` (libsbml only) + antimony, so neither calls the other. Reentrancy buys nothing —
and if one is ever introduced it is precisely the corruption this guard exists to stop (the
inner call's ``clearPreviousLoads``/``freeAll`` destroys the outer conversion's global state).
An RLock would let that through silently; a Lock deadlocks, which the runner's stall detector
surfaces as a visible bug rather than a wrong model.

The GIL does NOT make this unnecessary — the C extension gives no guarantee about its own global
tables across the multiple antimony calls one conversion makes.
"""

from __future__ import annotations

import threading

from petab.v1.models.sbml_model import SbmlModel

#: Exported: any code driving libantimony outside ``SbmlModel`` must hold this.
ANTIMONY_LOCK = threading.Lock()

_INSTALLED_FLAG = "_petab_mcp_antimony_lock_installed"


def install() -> None:
    """Idempotently wrap the two conversion entry points with the process-wide lock."""
    if getattr(SbmlModel, _INSTALLED_FLAG, False):
        return

    orig_to = SbmlModel.to_antimony
    orig_from = SbmlModel.from_antimony  # a @staticmethod: accessing it yields the plain function

    def to_antimony(self, *args, **kwargs):
        with ANTIMONY_LOCK:
            return orig_to(self, *args, **kwargs)

    def from_antimony(*args, **kwargs):
        with ANTIMONY_LOCK:
            return orig_from(*args, **kwargs)

    to_antimony.__doc__ = orig_to.__doc__
    from_antimony.__doc__ = orig_from.__doc__
    SbmlModel.to_antimony = to_antimony
    SbmlModel.from_antimony = staticmethod(from_antimony)
    setattr(SbmlModel, _INSTALLED_FLAG, True)
